import os
import re
import uuid
import asyncio
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from datasets import load_dataset

from model.vllm_engine import VllmEngine
from hparams import get_eval_args

app = FastAPI()
# Changes the process creation method to spawn to prevent conflicts between CUDA and multiprocessing.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def general_detokenize(string):
    """Detokenizes a string by correcting misplaced spaces"""
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def doc_to_text(doc):
    """Returns a prompt that fits the klue-sts dataset"""
    return f"""\n####\n제목: {doc["title"]}\n####\n내용: {doc["context"]}\n####\n문제: {doc["question"]}\n####\n"""


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/request")
async def generate_text(request: Request) -> Response:
    args = await request.json()

    # Converts the input dictionary into an argument class format
    model_args, data_args, eval_args, generation_args = get_eval_args(args)

    # Creates an instance of VllmEngine to initialize the model and generation parameters
    engine = VllmEngine(model_args, data_args, generation_args, eval_args)

    # Loads the dataset and generate the prompt
    data_files = {"validation": "/home/data/1_convert/korquad_dev/converted_dev2.json"}
    dataset = load_dataset("json", data_files=data_files)
    task_docs = list(dataset["validation"])[:10]
    task_docs_prompts = [doc_to_text(doc) for doc in task_docs]

    # VllmEngine processes all prompts asynchronously
    i = 0
    while task_docs_prompts or engine.model.has_unfinished_requests():
        print(i)
        if task_docs_prompts:
            prompt = task_docs_prompts.pop(0)
            # print(prompt)
            until = ["\n", "\u200b", "##"]
            stop_token_ids = []
            for u in until:
                stop_token_ids.extend(engine.tokenizer.encode(u)[1:])
            # stop_token_ids = list(set(stop_token_ids))

            # print(stop_token_ids)
            # logger.info(f"token_length: {len(self.tokenizer.encode(context))}")
            engine._add_request(
                "chatcmpl-{}".format(uuid.uuid4().hex),
                prompt,
                stop_token_ids=stop_token_ids,
            )

        generator = engine.model.step()
        for result in generator:
            print(result.outputs[0].text)
            if result.finished:
                generated_id = result.request_id
                generated_text = result.outputs[0].text
                print(f"Generated: {generated_text}")

        i += 1

    print("finished")

    # for prompt, result in zip(task_docs_prompts, results):
    #     print(f"Prompt: {prompt}")
    #     print(f"Generated: {result[0]}\n")
    results = []

    return JSONResponse(results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11181)
