import os
import io
import json
import requests
import numpy as np
import pandas as pd
from typing import Any, Dict, Union
from transformers import AutoModelForCausalLM

from config.path import (
    MODEL_DIR,
    SAVE_MODEL_DIR,
)
from config.url import API_URL
from config.dataset import EVAL_DATASET_METRIC


def read_file(data_type) -> Union[Dict, pd.DataFrame]:
    response = requests.get(API_URL + f"/data/{data_type}")
    if response.status_code == 200:
        if data_type == "leaderboard" or data_type == "model_info":
            data = pd.read_json(io.BytesIO(response.content))
        else:
            data = json.loads(response.content.decode("utf-8"))
    return data


def save_file(data_type: str, data: Dict) -> None:
    response = requests.post(
        API_URL + f"/data/{data_type}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )


def write_file(file_path, content) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=4)


def save_results_to_json(
    logger,
    task_name: str,
    content: Dict[str, any],
    save_dir: str,
):
    """
    결과 딕셔너리를 json 파일 형식으로 정해진 경로에 저장하는 함수
    """

    file_path = os.path.join(save_dir, f"{task_name}.json")

    # 결과 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 업데이트된 데이터를 JSON 파일에 저장
    write_file(file_path, content)

    logger.info(f"{task_name} > {file_path} 저장 완료")


def get_model_info(model_name: str):
    try:
        # 모델 로드 및 파라미터 수 계산
        df = read_file("model_info")
        if model_name in df["Model"].values:
            num_params = df[df["Model"] == model_name]["#Params(B)"].iloc[0]
        else:
            model_path = os.path.join(MODEL_DIR, model_name)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            num_params = round(
                sum(p.numel() for p in model.parameters()) / (1024**3), 2
            )

        with open(f"{MODEL_DIR}/{model_name}/config.json", "r") as f:
            config = json.load(f)

        model_info = {
            "#Params(B)": num_params,
            "Precision": config["torch_dtype"],
            "Architecture": config["architectures"][0],
            "hidden_size": config["hidden_size"],
            "seq_length": config["max_position_embeddings"],
            "num_layers": (
                config["num_layers"]
                if "num_layers" in config.keys()
                else config["num_hidden_layers"]
            ),
            "num_heads": config["num_attention_heads"],
        }

    except Exception as e:
        model_info = {"error": str(e)}

    return model_info


def renew_leaderboard(
    logger,
    model_name: str,
    results: Dict[str, Any],
) -> None:
    """
    리더보드 json 파일 갱신하는 함수
    """
    leaderboard = read_file("leaderboard")
    models = read_file("model_info")
    eval_results = read_file("eval_results")

    if model_name not in leaderboard["Model"].values:
        model_info = get_model_info(model_name)
        if "error" in model_info.keys():
            logger.error(
                f"[{model_name}] 모델 정보 불러오기 실패 : {model_info.pop('error')}"
            )
            model_info["#Params(B)"] = 0
        # leaderboard에 모델 추가
        leaderboard.loc[len(leaderboard)] = [model_name] + [np.nan] * (
            len(leaderboard.columns) - 1
        )
        leaderboard.loc[leaderboard["Model"] == model_name, "#Params(B)"] = model_info[
            "#Params(B)"
        ]
        # eval_results에 모델 추가
        eval_results[model_name] = {}
        # models에 모델 추가
        if (
            model_name.split("/")[0] != SAVE_MODEL_DIR.split("/")[-1]
        ):  # 튜닝 모델이 아닐 시 추가
            models.loc[len(models)] = [model_name] + [0] * (len(models.columns) - 1)
            for key, value in model_info.items():
                models.loc[models["Model"] == model_name, key] = value

    if model_name not in eval_results.keys():
        # eval_results에 모델 추가
        eval_results[model_name] = {}

    try:
        for task_name, result in results.items():
            if "error" in result.keys():
                continue
            leaderboard[task_name] = leaderboard[task_name].astype(str)
            leaderboard.loc[leaderboard["Model"] == model_name, task_name] = str(
                result[EVAL_DATASET_METRIC[task_name]]
            )
            eval_results[model_name][task_name] = result
            logger.info(f"[{model_name}]-[{task_name}] 리더보드 반영 완료")

    except Exception as e:
        logger.error(e)

    dataset_list = list(EVAL_DATASET_METRIC.keys())
    leaderboard[dataset_list] = (
        leaderboard[dataset_list].astype(str).apply(lambda col: col.str[:6])
    )
    save_file("eval_results", dict(eval_results))
    save_file("leaderboard", leaderboard.to_dict(orient="records"))
    save_file("model_info", models.to_dict(orient="records"))
