import os
import json
import torch
from typing import Dict, Any
import multiprocessing

from config.logger import setup_logger
from config.path import LOG_DIR
from eval import Evaluator


def eval(task_id: str, args: Dict[str, Any], queue: multiprocessing.Queue):
    """
    평가 수행 함수
    """
    # 환경 설정
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = args.pop("device", "0")
    if isinstance(device, list):
        device = ",".join(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if args["tensor_parallel_size"] > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    # logger 설정
    log_files = os.listdir(LOG_DIR)
    for log_file in log_files:
        if log_file.startswith("eval"):
            os.remove(f"{LOG_DIR}/{log_file}")
    logger = setup_logger(
        "leap",
        "eval.log",
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

    queue.put(result)
    logger.info(f"Task {task_id} 작업 완료")
    logger.info(f"result: {json.dumps(result, indent=2)}")
