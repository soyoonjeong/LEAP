import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer

from model.vllm_engine import VllmEngine
from hparams import get_eval_args

app = FastAPI()
# Changes the process creation method to spawn to prevent conflicts between CUDA and multiprocessing.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def make_prompt(sentence_pairs):
    prompt = ""
    # fewshot-context
    for title, expected in sentence_pairs[:-1]:
        prompt += f"제목:{title}\n정답:{expected}\n"
    # Add a sentence that needs to be answered correctly
    title, expected = sentence_pairs[-1]
    prompt += f"제목:{title}\n정답:"
    return prompt


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/generateText")
async def generate_text(request: Request) -> Response:
    args = await request.json()

    # Converts the input dictionary into an argument class format
    model_args, data_args, eval_args, generation_args = get_eval_args(args)

    # Creates an instance of VllmEngine to initialize the model and generation parameters
    engine = VllmEngine(model_args, generation_args, eval_args)
    model_config = await engine.model.get_model_config()
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)

    # Input examples
    sentence_pairs = [
        ("유튜브 내달 2일까지 크리에이터 지원 공간 운영", "생활문화"),
        ("내년부터 국가RD 평가 때 논문건수는 반영 않는다", "사회"),
        ("야외서 생방송 하세요…액션캠 전용 요금제 잇따라", "IT과학"),
        ("정상회담 D1 文대통령 취임 후 남북관계 주요 일지", "정치"),
        ("월드컵 태극전사 16강 전초기지 레오강 입성종합", "스포츠"),
        ("베트남 경제 고성장 지속…2분기 GDP 6.71% 성장", "세계"),
        ("다시 포효한 황의조 3년 만의 A매치 골 집중력 유지한...", "스포츠"),
    ]
    prompt = make_prompt(sentence_pairs)
    print(prompt)

    # Test with klue-ynat dataset
    # Tokenize the completion words and determine the number of tokens
    completions = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
    answer_tokens = {completion: [] for completion in completions}
    for answer in answer_tokens.keys():
        answer_tokens[answer] = tokenizer.encode(answer)
        print(f"{answer}: {answer_tokens[answer]}")
    answer_tokens_len = {
        completion: len(answer_tokens[completion]) for completion in completions
    }

    # Extract the logprobs of the prompt by entering the prompt in the vllm engine
    # Extract loglikelihood corresponding to the correct answer token by slicing the prompt's logprobs
    probabilities = {completion: float("-inf") for completion in completions}
    for completion in probabilities.keys():
        result_generator = await engine._generator(prompt + completion)
        async for result in result_generator:
            prompt_logprobs = result.prompt_logprobs[
                -answer_tokens_len[completion] + 1 :
            ]  # 맨 앞 토큰 제외
        print(prompt_logprobs)
        probability = 0
        for logprob_dict in prompt_logprobs:
            # 하나의 prompt_logprobs의 element(dictionary)에 두 개의 item이 있을 경우, 하나는 prompt에 이미 존재하는 token, 하나는 가장 logprob가 높은 토큰
            # 예시) {239012: Logprob(logprob=-inf, rank=3, decoded_token='같'), 236039: Logprob(logprob=-0.2840934097766876, rank=1, decoded_token='다')}
            min_prob = min(logprob_dict.items(), key=lambda x: x[1].logprob)
            probability += float(min_prob[1].logprob)
        if probability != float("-inf"):
            probabilities[completion] = probability
    print(probabilities)
    result = max(probabilities, key=probabilities.get)

    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11181)
