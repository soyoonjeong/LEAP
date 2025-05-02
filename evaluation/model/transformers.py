import os
import json
import collections
from typing import Optional, List, Union, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers
from accelerate import find_executable_batch_size

from . import utils
from hparams import (
    ModelArguments,
    GenerationArguments,
    EvaluationArguments,
)
from config import setup_logger


def get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


class TransformerEngine:
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
            model_args (ModelArguments): 모델 구성 및 파라미터
            data_args (DataArguments): 데이터 구성 및 파라미터
            generation_args (GenerationArguments): 생성 옵션 및 설정
            eval_args (EvaluationArguments): 평가 옵션 및 설정
        """
        self.generation_args = generation_args.to_dict()
        self.logger = logger

        # setup for automatic batch size detection
        self.batch_size = (
            int(eval_args.batch_size)
            if eval_args.batch_size != "auto"
            else eval_args.batch_size
        )
        self.max_new_tokens = generation_args.max_new_tokens
        self.max_model_len = model_args.max_model_len
        self.torch_dtype = (
            model_args.infer_dtype
            if model_args.infer_dtype == "auto"
            else getattr(torch, model_args.infer_dtype)
        )

        # 모델이 로컬에 저장되어 있는지 확인
        model_path = os.path.join(model_args.model_dir, model_args.model)
        if not os.path.exists(model_path):
            utils.download_model(
                model_args.model_dir, model_args.model
            )  # 모델이 로컬에 저장되어 있지 않으면 로컬에 다운로드
            logger.info(f"모델 {model_args.model}: 로컬에 다운로드 완료")

        # 토크나이저 로드
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model,
            trust_remote_code=True,
        )
        self.tokenizer.model_max_length = self.max_model_len

        # 모델 로드
        model_kwargs = {}
        if model_args.use_accelerate:
            model_kwargs = get_accelerate_args(
                model_args.device_map_option,
                model_args.max_memory_per_gpu,
                model_args.max_cpu_memory,
                model_args.offload_folder,
            )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            **model_kwargs,
        )
        self.model.eval()
        torch.set_grad_enabled(False)

        # 모델 디바이스 설정
        device_list = set(
            ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        if model_args.device and model_args.device in device_list:
            self.device = torch.device(model_args.device)
            self.logger.info(f"Using device '{model_args.device}'")
        else:
            self.logger.warning("Device not specified")
            self.logger.warning(f"Cuda Available? {torch.cuda.is_available()}")
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if not model_args.use_accelerate:
            self.model.to(self.device)

        self.tqdm_logger = setup_logger("tqdm", "eval_tqdm.log")

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

    def loglikelihood(self, requests: List[Tuple[str, Tuple[str, str]]]):
        new_reqs = []
        request_ids = []
        response_dict = collections.defaultdict(list)
        try:
            for request_id, request in requests:
                context, continuation = request.args
                if context == "":
                    # end of text as context
                    context_enc = [self.tokenizer.eos_token_id]
                else:
                    context_enc = self.tokenizer.encode(context)

                continuation_enc = self.tokenizer.encode(continuation)

                new_reqs.append(
                    ((context, continuation), context_enc, continuation_enc)
                )
                request_ids.append(request_id)

            result = self._loglikelihood_tokens(new_reqs)
        except Exception as e:
            self.logger.error(f"error: {e}")
            return e

        for request_id, (logprob, exact_match) in zip(request_ids, result):
            task_name, doc_id, i = request_id.split("/")
            response_dict[(task_name, int(doc_id))].append((i, logprob))

        return response_dict

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        """
        주어진 토큰 시퀀스에 대한 로그 가능도를 계산합니다.

        Args:
            requests (List): (context, continuation) 쌍을 포함하는 요청 리스트
            disable_tqdm (bool, optional): tqdm 진행 표시줄 비활성화 여부. 기본값은 False로 진행 표시줄 활성화
            override_bs (int, optional): 배치 크기를 수동으로 설정. 기본값은 None

        Returns:
            List: 각 요청에 대한 (로그 확률, 정확한 일치 여부) 튜플 리스트
        """
        res = []

        def _collate(x):
            """
            요청을 토큰의 길이가 긴 순서대로 정렬
            배치 처리 시 OOM 방지
            """
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        # 배치 크기 추정
        if len(re_ord.get_reordered()) > 0:
            _, context_enc, continuation_enc = re_ord.get_reordered()[0]
            max_context = len(
                (context_enc + continuation_enc)[-(self.max_model_len + 1) :][:-1]
            )
            if self.batch_size == "auto":
                if override_bs is None:
                    self.logger.info(
                        "Passed argument batch_size = auto. Detecting largest batch size"
                    )

                    # find_executable batch_size라는 데코레이터로 시작 배치 크기를 512로 설정하고
                    # OOM이 발생하면 배치 크기를 절반으로 줄여가며 적절한 배치 크기 찾음
                    @find_executable_batch_size(starting_batch_size=512)
                    def forward_batch(batch_size):
                        test_batch = torch.ones(
                            (batch_size, max_context), device=self.device
                        ).long()
                        if next(self.model.parameters()).device != self.device:
                            self.model.to(self.device)
                        for _ in range(5):
                            out = F.log_softmax(self.model(test_batch)[0], dim=-1).to(
                                self.device
                            )
                        return batch_size

                    batch_size = forward_batch()
                    self.logger.info(f"Determined largest batch size: {batch_size}")
                    adaptive_batch_size = batch_size

                else:  # override_bs가 존재하면 그 값을 사용
                    adaptive_batch_size = override_bs
        else:
            adaptive_batch_size = 0 if override_bs is None else override_bs

        # 요청을 배치로 나누어 처리
        pbar = tqdm(re_ord.get_reordered(), disable=disable_tqdm)
        for chunk in utils.chunks(
            pbar,
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # 각 요청에 대해 입력 시퀀스를 생성하고 패딩 길이를 추정
            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_model_len

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # 여러 시퀀스를 하나의 배치로 묶기 위해 입력 시퀀스를 패딩하여 동일한 길이로 만듦
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_model_len + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )
                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            # 모델을 통해 로그 확률을 계산
            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            multi_logits = F.log_softmax(
                self.model(batched_inps)[0], dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            # 각 요청에 대해 처리
            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # 각 요청에 대해 연속 토큰의 로그 확률을 추출
                contlen = len(cont_toks)
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # 그리디 디코딩 결과가 실제 연속 토큰과 정확히 일치하는지 확인
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # 결과 저장: (로그 확률, 정확한 일치 여부)
                answer = (float(logits.sum()), bool(max_equal))

                # if cache_key is not None:
                #     self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

            self.log_tqdm("", pbar)

        return re_ord.get_original(res)
