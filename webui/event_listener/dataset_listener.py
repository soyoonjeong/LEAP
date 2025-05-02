import os
import json
import time
import requests
import gradio as gr
import pandas as pd

from config.url import API_URL
from config.dataset import get_eval_datasets, get_tuning_datasets, DATASET_PATH
from config.path import SAVE_DATASET_DIR


def select_eval_dataset(dataset):
    # dataset_info
    EVAL_DATASETS = get_eval_datasets()
    info = EVAL_DATASETS[dataset]
    df = pd.DataFrame(info)
    dataset_info = gr.DataFrame(
        value=df, label="Information", visible=True, interactive=False
    )
    # dataset_preview
    data, subject = dataset.split("_")
    hf_iframe = f"""<iframe
                    src="https://huggingface.co/datasets/{DATASET_PATH[data]}/embed/viewer/{subject}/validation"
                    frameborder="0"
                    width="100%"
                    height="560px"
                    ></iframe>"""
    dataset_preview = gr.HTML(hf_iframe)

    return dataset_info, dataset_preview


def select_tuning_dataset(dataset):
    response = requests.get(API_URL + f"/dataset/sample/{dataset.replace('/', '$')}")
    data = response.json()
    return data["sample"]


def upload_tuning_dataset(file_name, file):
    TUNING_DATASETS = get_tuning_datasets()
    return_dropdown = gr.Dropdown(
        choices=TUNING_DATASETS,
        label="Dataset",
        visible=True,
        interactive=True,
        allow_custom_value=True,
    )
    # 파일 이름 입력 안한 경우
    if file_name == "":
        gr.Warning("파일 이름을 입력해주세요.")
        return return_dropdown, None

    # {"instruction", "output"} 형식이 아닌 경우
    with open(file.name, "r") as f:
        data = json.load(f)

    for item in data:
        if list(item.keys()) != ["instruction", "output"]:
            gr.Warning(f"{'Instruction', 'Output'} 형식이 아닙니다. \n {item}")
            return return_dropdown, None

    # 데이터셋 저장 경로에 저장 후 학습 데이터셋 리스트에 추가
    with open(file.name, "rb") as f:
        response = requests.post(
            API_URL + "/dataset/upload", files={"file": (file_name + ".json", f)}
        )
        TUNING_DATASETS[file_name] = os.path.join(SAVE_DATASET_DIR, file_name + ".json")
    if response.status_code != 200:
        gr.Warning("데이터셋 저장에 실패했습니다.")
    else:
        gr.Info("데이터셋 저장에 성공했습니다!")

    TUNING_DATASETS = get_tuning_datasets()
    return (
        gr.Dropdown(
            choices=TUNING_DATASETS,
            label="Dataset",
            visible=True,
            interactive=True,
            allow_custom_value=True,
        ),
        None,
    )
