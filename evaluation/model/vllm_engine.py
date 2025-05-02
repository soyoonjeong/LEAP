import collections
from typing import List, Tuple, Dict

from tqdm import tqdm
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

from hparams import (
    ModelArguments,
    GenerationArguments,
    EvaluationArguments,
)
from task.base import Request
from .vllm import Vllm


class VllmEngine(Vllm):
    def __init__(
        self,
        task_id: str,
        model_args: ModelArguments,
        eval_args: EvaluationArguments,
        generation_args: GenerationArguments,
        logger,
    ) -> None:
        """
        VllmEngine 초기화

        Args:
            task_id (str): scheduler에서 받아온 task_id
            model_args (ModelArguments): 모델 구성 및 파라미터
            generation_args (GenerationArguments): 생성 옵션 및 설정
            eval_args (EvaluationArguments): 평가 옵션 및 설정
            logger (logging.Logger): 평가 작업 수행 로그 기록
        """
        super().__init__(task_id, model_args, eval_args, generation_args, logger)

    def initialize_engine(self, engine_args) -> LLMEngine:
        """모델 로드"""
        return LLMEngine.from_engine_args(EngineArgs(**engine_args))

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
        id_target = {
            id: request.args[1] for id, request in requests
        }  # 요청 ID와 target 쌍 딕셔너리
        dictionary = self.get_dictionary(requests)

        # STEP 3: 필요한 자료구조 초기화 (e.g. 추론 결과 저장 딕셔너리)
        response_dict = collections.defaultdict(list)
        pbar = tqdm(requests, total=len(requests), desc="Description", leave=True)

        # STEP 4: 추론 수행
        try:
            # request가 남았거나 추론이 끝나기 전까지 반복
            while requests or self.model.has_unfinished_requests():
                # STEP 4-1: llm engine에 request 추가
                if requests:
                    request_id, single_request = requests.pop(0)
                    context, target = single_request.args

                    multi_modal_data = None
                    self.model.add_request(
                        inputs={
                            "prompt": context
                            + target,  # target에 대한 로그 확률을 계산할 것이기 때문에 context + target 전달
                            "multi_modal_data": multi_modal_data,
                        },
                        params=sampling_params,
                        request_id=request_id,  # 입력 순서 != 출력 순서
                    )

                # STEP 4-2: 추론
                request_outputs: List[RequestOutput] = self.model.step()
                for request_output in request_outputs:
                    generated_id = request_output.request_id
                    prompt_logprobs = request_output.prompt_logprobs[
                        -dictionary[id_target[generated_id]] :
                    ]  # target 토큰 길이만큼 자르기

                # STEP 4-3-1: 추론이 되었다면 target에 대한 로그 확률 계산
                if len(request_outputs) > 0:
                    logprob = 0
                    for logprob_dict in prompt_logprobs:
                        # 하나의 prompt_logprobs의 element(dictionary)에 두 개의 item이 있을 경우, 하나는 prompt에 이미 존재하는 token, 하나는 가장 logprob가 높은 토큰
                        # 예시) {239012: Logprob(logprob=-inf, rank=3, decoded_token='같'), 236039: Logprob(logprob=-0.2840934097766876, rank=1, decoded_token='다')}
                        min_prob = min(logprob_dict.items(), key=lambda x: x[1].logprob)
                        logprob += float(min_prob[1].logprob)

                    # 결과를 response_dict에 저장
                    task_name, doc_id, i = generated_id.split("/")
                    response_dict[(task_name, int(doc_id))].append((int(i), logprob))
                    pbar.update(1)
                    self.log_tqdm(task_name, pbar)

                # STEP 4-3-2: 추론이 되지 않았다면 다시 requests에 추가
                else:
                    requests.append((request_id, single_request))
                    pass
        except Exception as e:
            # 중간에 에러 발생 시 에러 딕셔너리 반환
            self.logger.error(f"""vLLM Inference Error : {e}""")
            pbar.close()
            return {"error": str(e)}

        pbar.close()
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
        request_id = 0
        pbar = tqdm(total=len(requests))
        response_dict = collections.defaultdict(list)
        task_name = requests[0][0].split("/")[0]
        prompts = [req.args[0] for _, req in requests]  # 프롬프트 리스트

        # STEP 3: 추론 수행
        try:
            # prompts가 남았거나 추론이 끝나기 전까지 반복
            while prompts or self.model.has_unfinished_requests():
                # STEP 4-1: llm engine에 prompt 추가
                if prompts:
                    prompt = prompts.pop(0)
                    self.model.add_request(
                        request_id=str(request_id),
                        inputs={
                            "prompt": prompt,
                            "multi_modal_data": None,
                        },
                        params=sampling_params,
                    )
                    request_id += 1

                # STEP 4-2: 추론
                request_outputs: List[RequestOutput] = self.model.step()
                for request_output in request_outputs:
                    if request_output.finished:
                        generated_id = request_output.request_id
                        generated_text = request_output.outputs[0].text
                        # STEP 4-3: 결과를 response_dict에 저장
                        response_dict[(task_name, int(generated_id))].append(
                            (0, generated_text)
                        )
                        pbar.update(1)
                        self.log_tqdm(task_name, pbar)
        except Exception as e:
            # 중간에 에러 발생 시 에러 딕셔너리 반환
            self.logger.error(f"""vLLM Inference Error : {e}""")
            pbar.close()
            return {"error": str(e)}

        pbar.close()
        return response_dict
