import json
import gradio as gr
from event_listener import dataset_listener


def create_tuning_dataset_tab(TUNING_DATASETS):
    with gr.Tab("📜 Tuning Dataset"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("<br> **지원하는 학습 데이터셋을 확인해보세요.**")
                tuning_dataset_select = gr.Dropdown(
                    TUNING_DATASETS,
                    label="Dataset",
                    visible=True,
                    interactive=True,
                    value="answerable/1_k_data_chunk_1.json",
                )
                gr.Markdown("**학습 데이터셋을 추가해보세요.** (json 파일만 가능)")
                file_name = gr.Textbox(
                    label="New Dataset Name",
                    visible=True,
                    interactive=True,
                )
                file_upload = gr.File(
                    label="Upload Dataset",
                    visible=True,
                    interactive=True,
                )
            tuning_dataset_preview = gr.JSON(scale=2, visible=True)

    tuning_dataset_select.change(
        fn=dataset_listener.select_tuning_dataset,
        inputs=[tuning_dataset_select],
        outputs=[tuning_dataset_preview],  # , tuning_dataset_delete_btn],
    )

    file_upload.upload(
        fn=dataset_listener.upload_tuning_dataset,
        inputs=[file_name, file_upload],
        outputs=[tuning_dataset_select, file_upload],
    )
