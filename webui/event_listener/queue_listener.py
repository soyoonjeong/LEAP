import gradio as gr
import pandas as pd

from .utils import create_args
from .status_listener import update_log, update_tuning_graph


def create_dropdown(elems):
    if len(elems) == 0:
        return gr.Dropdown(None, visible=True, interactive=False, scale=1)
    else:
        return gr.Dropdown(elems, visible=True, interactive=True, scale=1)


# Tasks 선택지 업데이트
def update_dropdown(queue):
    finished_queue_elems = []
    pending_queue_elems = []
    try:
        for id, task in dict(queue).items():
            task["id"] = id
            if task["status"] == "completed":
                finished_queue_elems.append(("-".join(id.split("-")[:2]), task))
            if task["status"] == "queued":
                pending_queue_elems.append(("-".join(id.split("-")[:2]), task))

    except Exception as e:
        print(f"Error in update_radio: {e}")

    finished_queue_radio = create_dropdown(finished_queue_elems)
    pending_queue_radio = create_dropdown(pending_queue_elems)

    return finished_queue_radio, pending_queue_radio


# Tasks 에서 선택 시 실행
def select_queue(queue_elems):
    # args
    args = create_args(queue_elems)
    # metrics
    try:
        if "error" in queue_elems["result"].keys():
            queue_elems["result"] = {"error": [queue_elems["result"]["error"]]}
        try:
            df = pd.DataFrame(queue_elems["result"]).T.reset_index()
        except:
            df = pd.DataFrame.from_dict(
                queue_elems["result"], orient="index"
            ).reset_index()
            df.columns = ["step", "value"]
        if "error" in queue_elems["result"].keys():
            df.columns = ["index", "value"]
        metrics = gr.DataFrame(
            value=df, label="Metrics", visible=True, interactive=False, max_height=250
        )
    except Exception as e:
        metrics = gr.DataFrame(visible=False, interactive=False)

    # cancel_btn
    cancel_btn = gr.Button("Cancel", visible=False, interactive=False)
    try:
        if queue_elems["status"] == "queued":
            cancel_btn = gr.Button("Cancel", visible=True, interactive=True)
    except:
        pass
    return args, metrics, cancel_btn


def select_tuning_history_queue(queue_elems):
    args, metrics, _ = select_queue(queue_elems)
    id = queue_elems["id"]
    log = update_log(id, id)
    loss_graph = update_tuning_graph(id, log, "Loss")
    grad_norm_graph = update_tuning_graph(id, log, "Grad Norm")

    return args, metrics, loss_graph, grad_norm_graph
