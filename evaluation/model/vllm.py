import os
import json
import collections
from typing import List, Tuple, Dict

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from hparams import (
    ModelArguments,
    GenerationArguments,
    EvaluationArguments,
)
from task.base import Request
from .utils import download_model
from config import setup_logger


class Vllm:
    def __init__(
        self,
        task_id: str,
        model_args: ModelArguments,
        eval_args: EvaluationArguments,
        generation_args: GenerationArguments,
        logger,
    ) -> None:
        """
        Vllm 초기화

        Args:
            task_id (str): scheduler에서 받아온 task_id
            model_args (ModelArguments): 모델 구성 및 파라미터
            generation_args (GenerationArguments): 생성 옵션 및 설정
            eval_args (EvaluationArguments): 평가 옵션 및 설정
            logger (logging.Logger): 평가 작업 수행 로그 기록
        """
        self.generation_args = generation_args.to_dict()  # 생성 옵션

        # 모델이 로컬에 저장되어 있는지 확인
        model_path = os.path.join(model_args.model_dir, model_args.model)
        if not os.path.exists(model_path):
            download_model(
                model_args.model_dir, model_args.model
            )  # 모델이 로컬에 저장되어 있지 않으면 로컬에 다운로드
            logger.info(f"모델 {model_args.model}: 로컬에 다운로드 완료")

        # 모델 설정
        engine_args = {
            "model": model_path,
            "trust_remote_code": True,
            "max_model_len": model_args.max_model_len,  # min(model_args.vllm_maxlen, data_args.cutoff_len),
            "tensor_parallel_size": model_args.tensor_parallel_size,  # torch.cuda.device_count() or 1,
            "gpu_memory_utilization": model_args.gpu_memory_utilization,
            "max_num_seqs": (
                64 if eval_args.batch_size == "auto" else int(eval_args.batch_size)
            ),
        }
        # 로깅 설정
        self.logger = logger
        self.tqdm_logger = setup_logger("tqdm", "eval_tqdm.log")
        self.logger.info(json.dumps(engine_args, indent=2))
        # 모델 및 토크나이저 로드
        self.model = self.initialize_engine(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, skip_special_tokens=True
        )

    def initialize_engine(self, engine_args) -> LLM:
        """모델 로드"""
        return LLM(**engine_args)

    def get_dictionary(self, requests):
        """target(대상) 텍스트에 대해서 토큰 길이 사전 생성"""
        targets = [request.args[1] for _, request in requests]
        # 토큰 사전에 등록
        token_dictionary = {
            target: self.tokenizer.encode(target) for target in list(set(targets))
        }
        # 모든 토큰 사전 요소들에 대해 첫 요소가 모두 같을 경우 첫 요소 제거
        first_elements = [tokens[0] for tokens in token_dictionary.values()]
        if len(set(first_elements)) == 1:
            len_dictionary = {
                target: len(tokens) - 1 for target, tokens in token_dictionary.items()
            }
        else:
            len_dictionary = {
                target: len(tokens) for target, tokens in token_dictionary.items()
            }

        return len_dictionary

    def log_tqdm(self, task_name, pbar):
        """tqdm 로그 기록 (GUI의 progress bar에 띄우기 위해)"""
        rate = pbar.format_dict["rate"]
        total = pbar.format_dict["total"]
        n = pbar.format_dict["n"]
        self.tqdm_logger.info(
            json.dumps(
                {
                    "task_name": task_name,
                    "current_steps": n,
                    "total_steps": total,
                    "elapsed_time": pbar.format_dict["elapsed"],
                    "remaining_time": (total - n) / rate if rate and total else 0,
                }
            )
        )

    def loglikelihood(
        self,
        requests: List[Tuple[str, Request]],
    ) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
        """
        Sampling params 내 prompt_logprobs 사용해서
        주어진 context와 target에 대해 target에 해당하는 log likelihood를 계산

        Args:
            requests (List[Tuple[str, Request]]):  요청 ID와 인수를 포함하는 요청 목록
            request_id, (context, target) 형식으로 구성됨
            request_id는 요청을 식별하기 위한 고유 ID
            context는 모델에 제공할 프롬프트, target은 log likelihood를 계산할 목표 텍스트

        Returns:
            response_dict (Dict[Tuple[str, int], List[Tuple[int, float]]]): 결과를 저장할 딕셔너리
        """
        # STEP 1: 생성 옵션 설정
        sampling_params = SamplingParams(
            seed=42,
            temperature=self.generation_args["temperature"],
            top_p=self.generation_args["top_p"],
            top_k=self.generation_args["top_k"],
            max_tokens=1,  # target에 대한 로그 확률만 계산할 것이기 때문에 생성은 필요 없음
            prompt_logprobs=1,  # 프롬프트의 로그 확률을 계산하기 위한 옵션
        )
        self.logger.info(sampling_params)

        # STEP 2: target 토큰화해서 토큰 길이 사전 생성 (e.g. {"대한민국": 2, "한국": 1})
        dictionary = self.get_dictionary(requests)

        # STEP 3: 필요한 자료구조 초기화 (e.g. 추론 결과 저장 딕셔너리)
        response_dict = collections.defaultdict(list)
        pbar = tqdm(requests, total=len(requests), desc="Description", leave=True)

        # STEP 4: 추론 수행
        try:
            for request_id, single_request in pbar:
                context, target = single_request.args

                # STEP 4-1. 추론
                output = self.model.generate(
                    prompts=context
                    + target,  # target에 대한 로그 확률을 계산할 것이기 때문에 context + target 전달
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                prompt_logprobs = output[0].prompt_logprobs[
                    -dictionary[target] :
                ]  # target 토큰 길이만큼 자르기

                # STEP 4-2. target에 해당하는 토큰의 로그 확률 합 계산
                logprob = 0
                for logprob_dict in prompt_logprobs:
                    # 하나의 prompt_logprobs의 element(dictionary)에 두 개의 item이 있을 경우, 하나는 prompt에 이미 존재하는 token, 하나는 가장 logprob가 높은 토큰
                    # e.g., {239012: Logprob(logprob=-inf, rank=3, decoded_token='같'), 236039: Logprob(logprob=-0.2840934097766876, rank=1, decoded_token='다')}
                    min_prob = min(logprob_dict.items(), key=lambda x: x[1].logprob)
                    logprob += float(min_prob[1].logprob)

                # STEP 4-3. 결과를 response_dict에 저장
                task_name, doc_id, i = request_id.split("/")
                response_dict[(task_name, int(doc_id))].append((int(i), logprob))
                self.log_tqdm(task_name, pbar)  # tqdm 로그 기록

        except Exception as e:
            # 추론 중간에 에러 발생 시 에러 딕셔너리 반환
            self.logger.error(f"""vLLM Inference Error : {e}""")
            return {"error": str(e)}

        return response_dict

    def greedy_until(
        self,
        requests: List[Tuple[str, Request]],
    ) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
        """
        주어진 until 단어 또는 문장이 나타날 때까지 모델이 생성하는 텍스트를 반환

        Args:
            requests (List[Tuple[str, Request]]):  요청 ID와 인수를 포함하는 요청 목록
            request_id, (prompt, {"until": [...]}) 형식으로 구성됨
            request_id는 요청을 식별하기 위한 고유 ID
            prompt는 모델에 제공할 프롬프트, until은 모델이 생성을 중지할 조건을 나타내는 목록

        Returns:
            response_dict (Dict[Tuple[str, int], List[Tuple[int, float]]]): 결과를 저장할 딕셔너리
        """
        # STEP 1: 생성 옵션 설정
        sampling_params = SamplingParams(
            temperature=self.generation_args["temperature"],
            top_p=self.generation_args["top_p"],
            top_k=self.generation_args["top_k"],
            stop=requests[0][1].args[1]["until"],
        )
        self.logger.info(sampling_params)

        # STEP 2: 필요한 자료구조 초기화 (e.g. 추론 결과 저장 딕셔너리)
        response_dict = collections.defaultdict(list)
        task_name = requests[0][0].split("/")[0]
        prompts = [req.args[0] for _, req in requests]  # 프롬프트 리스트 생성

        # STEP 3: 추론 수행
        try:
            # STEP 3-1: 추론
            outputs = self.model.generate(prompts, sampling_params)
            pbar = tqdm(outputs, total=len(requests), desc="Description", leave=True)
            for output in pbar:
                generated_text = output.outputs[0].text
                generated_id = output.request_id
                # STEP 3-2: 결과를 response_dict에 저장
                response_dict[(task_name, int(generated_id))].append(
                    (0, generated_text)
                )
                self.log_tqdm(task_name, pbar)
        except Exception as e:
            # 중간에 에러 발생 시 에러 딕셔너리 반환
            self.logger.error(f"""vLLM Inference Error : {e}""")
            return {"error": str(e)}

        return response_dict
