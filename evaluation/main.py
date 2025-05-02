import os
import fnmatch
import argparse
import torch
import multiprocessing

from eval import Evaluator
from task import ALL_TASKS

from config.logger import setup_logger


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--task", type=str, required=True, choices=MultiChoice(ALL_TASKS)
    )
    parser.add_argument("--model_dir", type=str, default="/home")
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--infer_backend", type=str, default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_fewshot", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="/home/leap/eval_result")
    parser.add_argument("--write_out", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    args = dict(args.__dict__)

    # 환경 설정
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = args.pop("device", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if args["tensor_parallel_size"] > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    # logger 설정
    task_id = args.pop("task_id", "0")
    logger = setup_logger(
        "leap",
        f"{task_id}.log",
    )

    # 평가 수행
    logger.info(f"Task {task_id} 작업 시작")
    try:
        # Evaluator 객체 생성 및 평가 수행
        evaluator = Evaluator(task_id, args, logger)
        result = evaluator.eval()
    except Exception as e:
        # 평가 중 오류 발생 시 오류 메시지 반환
        logger.error(e)
        result = {"error": str(e)}

    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()
    result = main()
