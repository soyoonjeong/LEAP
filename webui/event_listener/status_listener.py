import os
import re
import json
import random
import datetime
import requests
import collections

import gradio as gr
import matplotlib.pyplot as plt

from .utils import create_args

from config.path import LOG_DIR
from config.url import API_URL


# 0.1초마다 계속 반환값이 바뀌는 함수
def trigger_change():
    return round(random.uniform(0, 1), 5)


def update_log(current_task, type):
    log = "Not Running!"
    file_path = os.path.join(LOG_DIR, f"{type}.log")

    if current_task != "Not Running" and os.path.exists(file_path):
        with open(file_path, "r") as f:
            log = f.read()

    return log


def update_eval_pbar(current_task):
    pbar_label = "Not Running"
    pbar_value = 0
    file_path = os.path.join(LOG_DIR, f"eval_tqdm.log")

    if current_task != "Not Running" and os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.readlines()
            if len(content) > 0:
                latest_log = json.loads(content[-1])
                pbar_value = (
                    int(latest_log["current_steps"] / latest_log["total_steps"] * 100)
                    + 1
                )
                pbar_label = "[{}] Running {:d}/{:d}: {} < {}".format(
                    latest_log["task_name"],
                    latest_log["current_steps"],
                    latest_log["total_steps"],
                    str(datetime.timedelta(seconds=int(latest_log["elapsed_time"]))),
                    str(datetime.timedelta(seconds=int(latest_log["remaining_time"]))),
                )

    pbar = gr.Slider(
        label=pbar_label, value=pbar_value, visible=True, interactive=False
    )
    return pbar


def update_eval_display(queue):
    current = "Not Running"
    for id, task in queue.items():
        if task["status"] == "running":
            current = id
    # running_args 업데이트
    elems = queue[current] if current != "Not Running" else {}
    running_args = create_args(elems)

    # log 업데이트
    log = update_log(current, "eval")
    # pbar 업데이트
    pbar = update_eval_pbar(current)

    return (current, running_args, log, pbar)


def plot_graph(label, value, epochs):
    color = {"Loss": "red", "Grad Norm": "blue"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, value, label=label, marker="o", color=color[label])
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig  # Figure 객체를 반환


def update_tuning_graph(current_task, log, label):
    # 정규식 패턴 설정
    metrics_pattern = r"'loss': ([\d.e-]+), 'grad_norm': ([\d.e-]+), 'learning_rate': ([\d.e-]+), 'epoch': ([\d.e-]+)"

    if current_task != "Not Running":
        # 그래프 그리기
        matches = re.findall(metrics_pattern, log)
        values = collections.defaultdict(list)
        for match in matches:
            values["Loss"].append(float(match[0]))
            values["Grad Norm"].append(float(match[1]))
            values["Learning Rate"].append(float(match[2]))
            values["Epoch"].append(float(match[3]))
        fig = plot_graph(label, values[label], values["Epoch"])
        return fig
    else:
        return None


def update_tuning_pbar(current_task, log):
    # 정규식 패턴 설정
    tqdm_patterns = [
        r"\s(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+),\s+([\d.]+s/it)\]",
        r"\s(\d+)/(\d+) \[(\d+:\d+:\d+)<(\d+:\d+),\s+([\d.]+s/it)\]",
        r"\s(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+:\d+),\s+([\d.]+s/it)\]",
        r"\s(\d+)/(\d+) \[(\d+:\d+:\d+)<(\d+:\d+:\d+),\s+([\d.]+s/it)\]",
    ]

    pbar_label = "Not Running"
    pbar_value = 0

    if current_task != "Not Running":
        matches = []
        for pattern in tqdm_patterns:
            matches.extend(re.findall(pattern, log))

        if len(matches) > 0:
            matches = sorted(matches, key=lambda x: int(x[0]))
            tqdm_match = matches[-1]
            current, total, elapsed, remaining, speed = tqdm_match
            current = int(current)
            total = int(total)
            pbar_value = int(current / total * 100)
            # print(current, total, elapsed, remaining, speed)
            pbar_label = "[{}] Running {:d}/{:d}: {} < {}, {}".format(
                "run_tuning", current, total, elapsed, remaining, speed
            )

    pbar = gr.Slider(
        label=pbar_label, value=pbar_value, visible=True, interactive=False
    )
    return pbar


def update_tuning_display(queue):
    current = "Not Running"
    for id, task in queue.items():
        if task["status"] == "running":
            current = id
    # running_args 업데이트
    elems = queue[current] if current != "Not Running" else {}
    running_args = create_args(elems)

    # log 업데이트
    log = update_log(current, current)
    pbar = update_tuning_pbar(current, log)
    loss_graph = update_tuning_graph(current, log, "Loss")
    grad_norm_graph = update_tuning_graph(current, log, "Grad Norm")
    return (current, running_args, log, pbar, loss_graph, grad_norm_graph)


# log, pbar, queue, gpu_status 업데이트 (0.1초마다 한 번씩 실행됨)
def update_display():
    response = requests.get(API_URL + "/status")
    queue = response.json()
    gpu_status = queue.pop("gpu_status")
    eval_queue = {
        key: value for key, value in queue.items() if value["type"] == "evaluate"
    }
    tuning_queue = {
        key: value for key, value in queue.items() if value["type"] == "tuning"
    }
    eval_current, eval_running_args, eval_log, eval_pbar = update_eval_display(
        eval_queue
    )
    (
        tuning_current,
        tuning_running_args,
        tuning_log,
        tuning_pbar,
        loss_graph,
        grad_norm_graph,
    ) = update_tuning_display(tuning_queue)

    return (
        eval_current,
        eval_queue,
        eval_running_args,
        eval_log,
        eval_pbar,
        tuning_current,
        tuning_queue,
        tuning_running_args,
        tuning_log,
        tuning_pbar,
        loss_graph,
        grad_norm_graph,
        gpu_status,
        gpu_status,
    )
