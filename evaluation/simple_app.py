import os
import time
import json
import datetime

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from eval import Evaluator

from config.logger import setup_logger

app = FastAPI()
# Changes the process creation method to spawn to prevent conflicts between CUDA and multiprocessing.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
torch.cuda.memory._set_allocator_settings("expandable_segments:False")
# os.environ['VLLM_USE_MANAGED_MEMORY']= '1'
# os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['NCCL_DEBUG']='TRACE'
# os.environ['VLLM_TRACE_FUNCTION'] = '1'


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/request")
async def generate_text(request: Request) -> Response:
    args = await request.json()

    start = time.time()
    torch.cuda.empty_cache()
    now = datetime.datetime.now()
    model_name = args["model"].split("/")[-1]
    task_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{model_name}"

    logger = setup_logger(
        "leap",
        f"{task_id}.log",
    )

    evaluator = Evaluator(task_id, args, logger)
    result = evaluator.eval()
    torch.cuda.empty_cache()
    end = time.time()
    times = str(datetime.timedelta(seconds=end - start))
    short = times.split(".")[0]

    result["time"] = short
    (f"result: {json.dumps(result, indent=2)}")
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11181)
