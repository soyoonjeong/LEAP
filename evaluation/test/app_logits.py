import os
import copy
import uvicorn
import torch
import torch.nn.functional as F
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer
from typing import List

from model.vllm_engine import VllmEngine
from hparams import get_eval_args

app = FastAPI()
# Changes the process creation method to spawn to prevent conflicts between CUDA and multiprocessing.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/generateText")
async def generate_text(request: Request) -> Response:
    args = await request.json()

    # Converts the input dictionary into an argument class format
    model_args, data_args, eval_args, generation_args = get_eval_args(args)

    # Creates an instance of VllmEngine to initialize the model and generation parameters
    engine = VllmEngine(model_args, data_args, generation_args, eval_args)
    model_config = await engine.model.get_model_config()
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)

    # Input examples
    prompt = "유튜브 내달 2일까지 크리에이터 지원 공간 운영 (생활)\n\n네이버 아이디로 로그인 사용자 월 1천만명 넘어 (경제)\n\n일자리·생활혁신…정부 4차산업혁명 대응책 마련한다 (사회)\n\n미스터올스타 넥센 김하성 좋은 기운 쭉 이어지길종합 (스포츠)\n\n방정오 측 故장자연과 통화 보도 사실무근…법적대응 (사회)\n\n갤럭시 노트9 공개 발표하는 고동진 IM부문장 (과학)\n\n영상 사상 첫 온라인 개학…가보지 않은 길 새로운 도전 "
    print(prompt)

    # Test with klue-ynat dataset
    # Tokenize the completion words and determine the number of tokens
    completions = [
        "(과학)",
        "(경제)",
        "(사회)",
        "(생활)",
        "(세계)",
        "(스포츠)",
        "(정치)",
    ]
    target = "(경제)"

    answer_tokens = {completion: [] for completion in completions}
    for answer in answer_tokens.keys():
        answer_tokens[answer] = tokenizer.encode(answer)[1:]
        print(f"{answer}: {answer_tokens[answer]}")

    # logits_processor에서 log probabilties 계산 -> 느림
    # def custom_logits_processor(
    #     token_ids: List[int], logits: torch.Tensor
    # ) -> torch.Tensor:
    #     idx = len(token_ids)
    #     if idx < len(answer_tokens[target]):
    #         log_softmax = F.log_softmax(logits, dim=-1)
    #         target_logprobs.append(
    #             log_softmax[answer_tokens[target][idx]].detach().cpu()
    #         )
    #         print(
    #             f"{answer_tokens[target][idx]}: {log_softmax[answer_tokens[target][idx]]}"
    #         )
    #         logits[answer_tokens[target][idx]] = max(logits) + 10
    #     return logits
    def custom_logits_processor(
        token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        idx = len(token_ids)
        if idx < len(answer_tokens[target]):
            logits_list.append(copy.copy(logits))
            logits[answer_tokens[target][idx]] = logits.max() + 10
        return logits

    logits_list = []

    # Extract the logprobs of the prompt by entering the prompt in the vllm engine
    # Extract loglikelihood corresponding to the correct answer token by slicing the prompt's logprobs
    request_id = "TEST"
    generator = await engine._generator(
        request_id,
        prompt,
        logits_processors=[custom_logits_processor],
        max_new_tokens=len(answer_tokens[target]),
    )

    async for result in generator:
        if result.outputs[0].finish_reason != None:
            logits_tensor = torch.stack(logits_list)
            print(f"logits_tensor: {logits_tensor.shape}")
            log_probs = F.log_softmax(logits_tensor, dim=-1).cpu().unsqueeze(0)
            print(f"logits_tensor: {logits_tensor.shape}")
            target_token_tensor = torch.tensor(answer_tokens[target]).unsqueeze(0)
            print(f"target_token_tensor: {target_token_tensor}")
            log_probs_for_target = torch.gather(
                log_probs, 2, target_token_tensor.unsqueeze(-1)
            ).squeeze(-1)
            # log_probs = F.log_softmax(logits_tensor, dim=-1).cpu().unsqueeze(0)
            # target_token_tensor = (
            #     torch.tensor(answer_tokens[target]).unsqueeze(-1).to(log_probs.device)
            # )
            # log_probs_for_target = log_probs.gather(
            #     dim=-1, index=target_token_tensor
            # ).squeeze(-1)
            print(log_probs_for_target)
            print(result.outputs[0])
            logprob = log_probs_for_target.sum().item()

    result = {"result": logprob}
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11181)
