import gradio as gr

from gradio_modal import Modal
from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter
from event_listener import leaderboard_listener


from config.leaderboard import MODEL_ON_LOAD_COLUMNS, MODEL_TYPES
from config.guide import MEMORY_GUIDE


def create_gpu_tab():
    with gr.Tab("🤖 GPU Memory Estimation") as gpu_tab:
        with gr.Row():
            data_type = gr.Dropdown(
                label="Gradient Type",
                choices=["fp16", "fp32", "bf16", "int8", "int4"],
                value="fp32",
                interactive=True,
            )
            optimizer = gr.Dropdown(
                label="Optimizer",
                choices=["adamw", "sgd"],
                value="adamw",
                interactive=True,
            )
            activation_checkpointing = gr.Dropdown(
                label="Activation Checkpointing",
                choices=["full", "selective", "none"],
                value="full",
                interactive=True,
            )
            batch_size = gr.Number(label="Batch Size", value=1, interactive=True)
            tensor_parallel_size = gr.Number(
                label="Tensor Parallel Size",
                value=1,
                interactive=True,
            )
            with gr.Column():
                calculate_btn = gr.Button(
                    value="Calculate Training Memory", elem_classes="red_btn"
                )
                memory_guide_btn = gr.Button(value="Calculating Memory Guide")
            with Modal(visible=False) as guide_modal:
                gr.Markdown(MEMORY_GUIDE)
        df = leaderboard_listener.init_model_leaderboard()
        models = Leaderboard(
            value=df,
            select_columns=SelectColumns(
                default_selection=MODEL_ON_LOAD_COLUMNS,
                cant_deselect=[],
                label="Select Columns to Display:",
            ),
            search_columns=["Model"],
            filter_columns=[
                "Precision",
                ColumnFilter("#Params(B)", default=[0.1, 73]),
            ],
            datatype=MODEL_TYPES,
            # column_widths=["2%", "33%"],
        )

    gpu_tab.select(
        fn=leaderboard_listener.update_leaderboard, inputs=[models], outputs=[models]
    )
    calculate_btn.click(
        fn=leaderboard_listener.init_model_leaderboard,
        inputs=[
            data_type,
            optimizer,
            activation_checkpointing,
            batch_size,
            tensor_parallel_size,
        ],
        outputs=[models],
    )
    memory_guide_btn.click(
        fn=lambda: Modal(visible=True),
        inputs=[],
        outputs=[guide_modal],
    )
