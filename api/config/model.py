import os
import pandas as pd
from .path import MODEL_INFO_PATH, SAVE_MODEL_DIR


def list_folders(directory):
    """
    주어진 디렉토리 아래의 모든 폴더 이름을 반환합니다.

    Args:
        directory (str): 검색할 디렉토리 경로

    Returns:
        list: 폴더 이름의 리스트
    """
    folders = []
    try:
        for item in os.listdir(directory):
            if item.endswith("_Merge"):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    folders.extend(
                        [
                            os.path.join(
                                "llm-tuning-model", item_path.split("/")[-1], checkpoint
                            )
                            for checkpoint in os.listdir(item_path)
                        ]
                    )
    except:
        pass
    return folders


def get_eval_models():
    model_info = pd.read_json(MODEL_INFO_PATH)
    EVAL_MODELS = {}
    EVAL_MODELS["baseline"] = list(model_info["Model"])
    EVAL_MODELS["finetuned"] = list_folders(
        SAVE_MODEL_DIR
    )  # 튜닝 모델을 평가 모델로 추가
    return EVAL_MODELS


def get_tuning_models():
    TUNING_MODELS = {
        "tuning": [
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B",
            "meta-llama/Llama-3.1-8B",
            "google/gemma-1.1-2b-it",
            "google/gemma-1.1-7b-it",
            "google/gemma-2-2b",
            "google/gemma-2-9b",
        ]
    }
    return TUNING_MODELS
