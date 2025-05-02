import json
import time
import argparse
import requests

url = "http://0.0.0.0:11181/request"

headers = {"Content-Type": "application/json"}

models = [
    "/home/llm_models/Ko-Mixtral-v1.4-MoE-7Bx2",
    "/home/llm_models/ko-beomi/beomi-gemma-ko-7b",
]
model = "beomi/gemma-ko-7b"

# For testing purposes, specify the model, dataset, and generation options
# cutoff length, max_samples, batch_size, maximum new tokens, top_p, temperature

parser = argparse.ArgumentParser(description="Evaluate model performance")
parser.add_argument(
    "-m",
    "--model_name_or_path",
    type=str,
    help="Model name or path",
    default="/home/gemma-2-9b",
)
parser.add_argument(
    "-ib", "--infer_backend", type=str, help="Inference backend", default="vllm"
)
parser.add_argument("-t", "--task", type=str, help="Task name", default="korquad")
parser.add_argument(
    "-td", "--task_dir", type=str, help="Task directory", default="/home/data/1_convert"
)
parser.add_argument("-ns", "--n_shot", type=int, help="Number of few-shots", default=0)
parser.add_argument(
    "-mt", "--max_new_tokens", type=int, help="Maximum new tokens", default=1024
)
parser.add_argument("-lp", "--logprobs", type=int, help="Log probabilities", default=0)
parser.add_argument(
    "-plp", "--prompt_logprobs", type=int, help="Prompt log probabilities", default=0
)
parser.add_argument(
    "-tps", "--tensor_parallel_size", type=int, help="Tensor parallel size", default=2
)
parser.add_argument(
    "-mct",
    "--max_concurrent_tasks",
    type=int,
    help="Maximum concurrent tasks",
    default=1,
)
args = parser.parse_args()
args = dict(args.__dict__)
print(json.dumps(args, indent=2))

start_time = time.time()
response = requests.post(url, headers=headers, data=json.dumps(args))
end_time = time.time()
latency = end_time - start_time
print(f"Response : {response.text}, Latency : {latency}sec")
