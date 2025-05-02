import io
import requests
import numpy as np
import pandas as pd


from config.dataset import get_eval_datasets
from config.url import API_URL
from gpu_memory_compute import get_training_memory, get_inference_memory


def init_llm_leaderboard():
    response = requests.get(API_URL + "/data/leaderboard")
    if response.status_code == 200:
        df = pd.read_json(io.BytesIO(response.content))
        # Average 계산
        EVAL_DATASETS = get_eval_datasets()
        datasets = list(EVAL_DATASETS.keys())
        x_data = df[datasets]
        avg_data = np.mean(x_data, axis=1)  # 단순 평균
        avg_data = np.array(list(map(lambda x: round(x * 100, 2), avg_data)))
        df.insert(1, "Average ⬆️", avg_data)
        # Average로 정렬
        df = df.sort_values(by="Average ⬆️", ascending=False)
    else:
        df = pd.DataFrame()

    return df


def init_model_leaderboard(
    gradient: str = "fp32",
    optimizer: str = "adamw",
    activation_checkpointing: str = "full",
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
):
    response = requests.get(API_URL + "/data/model_info")
    if response.status_code == 200:
        models = pd.read_json(io.BytesIO(response.content))
        models = models.to_dict()
        new_models = models.copy()
        for idx, model_name in models["Model"].items():
            num_params = models["#Params(B)"][idx]
            precision = models["Precision"][idx]
            hidden_size = models["hidden_size"][idx]
            seq_length = models["seq_length"][idx]
            num_layers = models["num_layers"][idx]
            num_heads = models["num_heads"][idx]

            inference_mem = get_inference_memory(num_params=num_params)
            model_mem, optimizer_mem, gradient_mem, activation_mem = (
                get_training_memory(
                    model=model_name,
                    num_params=num_params,
                    gradient=gradient,
                    optimizer=optimizer,
                    activation_checkpointing=activation_checkpointing,
                    batch_size=batch_size,
                    tensor_parallel_size=tensor_parallel_size,
                    model_info={
                        "Precision": precision,
                        "hidden_size": hidden_size,
                        "seq_length": seq_length,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                    },
                )
            )
            new_models["inference_memory(GB)"][idx] = round(inference_mem, 2)
            new_models["model_memory(GB)"][idx] = round(model_mem, 2)
            new_models["optimizer_memory(GB)"][idx] = round(optimizer_mem, 2)
            new_models["gradient_memory(GB)"][idx] = round(gradient_mem, 2)
            new_models["activation_memory(GB)"][idx] = round(activation_mem, 2)
            new_models["training_memory(GB)"][idx] = round(
                model_mem + optimizer_mem + gradient_mem + activation_mem, 2
            )
    else:
        new_models = None
    return pd.DataFrame(new_models)


def update_leaderboard(leaderboard):
    if "inference_memory(GB)" not in leaderboard.columns.tolist():
        df = init_llm_leaderboard()
    else:
        df = init_model_leaderboard()

    return df
