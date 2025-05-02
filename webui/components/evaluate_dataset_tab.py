import gradio as gr
import pandas as pd
from event_listener import dataset_listener
from config.dataset import DATASET_PATH


def create_evaluate_dataset_tab(EVAL_DATASETS):
    with gr.Tab("📜 Evaluation Dataset"):
        with gr.Row():
            eval_dataset_select = gr.Dropdown(
                EVAL_DATASETS,
                label="Dataset",
                visible=True,
                interactive=True,
                value="klue_ynat",
            )
            eval_dataset_info = gr.DataFrame(
                value=pd.DataFrame(EVAL_DATASETS["klue_ynat"]),
                scale=2,
                visible=True,
                interactive=False,
            )
        hf_iframe = f"""<iframe
                        src="https://huggingface.co/datasets/{DATASET_PATH["klue"]}/embed/viewer/ynat/validation"
                        frameborder="0"
                        width="100%"
                        height="560px"
                        ></iframe>"""
        eval_dataset_preview = gr.HTML(hf_iframe)

    eval_dataset_select.change(
        fn=dataset_listener.select_eval_dataset,
        inputs=[eval_dataset_select],
        outputs=[eval_dataset_info, eval_dataset_preview],
    )
