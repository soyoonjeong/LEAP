import os
import gradio as gr

from css import css
from components import (
    tuning_tab,
    evaluate_tab,
    leaderboard_tab,
    tuning_dataset_tab,
    tuning_history_tab,
    evaluate_dataset_tab,
    gpu_tab,
)
from event_listener import status_listener, queue_listener
from config import (
    get_eval_datasets,
    get_tuning_datasets,
    get_eval_models,
    get_tuning_models,
)


def build_demo():
    with gr.Blocks(title="SURROMIND", css=css) as demo:
        gr.Markdown("# Surro LLM Eval & PEFT <br><br>")
        gr.DownloadButton(
            label="Download Manual PDF",
            value="/home/leap/webui/LEAP_기능설명서.pdf",
            visible=True,
            elem_classes=["download-btn"],
        )
        EVAL_DATASETS = get_eval_datasets()
        TUNING_DATASETS = get_tuning_datasets()
        EVAL_MODELS = get_eval_models()
        TUNING_MODELS = get_tuning_models()
        trigger = gr.Textbox(
            value=status_listener.trigger_change, visible=False, every=0.2
        )
        leaderboard_tab.create_leaderboard_tab()
        evaluate = evaluate_tab.create_evaluate_tab(EVAL_MODELS, EVAL_DATASETS)
        evaluate_dataset_tab.create_evaluate_dataset_tab(EVAL_DATASETS)
        tuning = tuning_tab.create_tuning_tab(TUNING_MODELS, TUNING_DATASETS)
        tuning_dataset_tab.create_tuning_dataset_tab(TUNING_DATASETS)
        tuning_history = tuning_history_tab.create_tuning_history_tab()
        gpu_tab.create_gpu_tab()

        # Queue Management
        trigger.change(
            fn=status_listener.update_display,
            inputs=[],
            outputs=[
                evaluate["current_task"],
                evaluate["queue"],
                evaluate["running_args"],
                evaluate["output_box"],
                evaluate["progress_bar"],
                tuning["current_task"],
                tuning["queue"],
                tuning["running_args"],
                tuning["output_box"],
                tuning["progress_bar"],
                tuning["loss_graph"],
                tuning["grad_norm_graph"],
                evaluate["gpu_status"],
                tuning["gpu_status"],
            ],
            show_progress=False,
        )
        # Tuning History
        tuning["tuning_tab"].select(
            fn=queue_listener.update_dropdown,
            inputs=[tuning["queue"]],
            outputs=[
                tuning_history["finished_queue_elems"],
                tuning["pending_queue_elems"],
            ],
            show_progress=False,
        )
        tuning_history["tuning_history_tab"].select(
            fn=queue_listener.update_dropdown,
            inputs=[tuning["queue"]],
            outputs=[
                tuning_history["finished_queue_elems"],
                tuning["pending_queue_elems"],
            ],
            show_progress=False,
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=64).launch(
        share=False,
        server_name="0.0.0.0",
        debug=True,
        server_port=int(os.getenv("FRONTEND_PORT", 11191)),
        allowed_paths=["/home/leap/webui/LEAP_기능설명서.pdf"],
    )
