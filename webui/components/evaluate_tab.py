import gradio as gr

from gradio_modal import Modal
from event_listener import (
    model_listener,
    queue_listener,
    request_listener,
    gpu_listener,
)

from config.guide import EVAL_USAGE_GUIDE


def create_evaluate_tab(EVAL_MODELS, EVAL_DATASETS):
    with gr.Tab("🚀 Evaluate") as evaluate_tab:
        queue = gr.JSON(label="Queue", visible=False)
        gpu_status = gr.JSON(label="GPU Status", visible=False)
        with gr.Column():
            with gr.Accordion(
                "✅ Finished Evaluation Queue",
                open=False,
            ):
                with gr.Row():
                    finished_queue_elems = gr.Dropdown(
                        show_label=False,
                        visible=True,
                        interactive=True,
                    )
                    with gr.Column(scale=6):
                        finished_args = gr.DataFrame(visible=False, interactive=False)
                        finished_metrics = gr.DataFrame(
                            visible=False, interactive=False
                        )
            with gr.Accordion(
                "⏳ Pending Evaluation Queue",
                open=False,
            ):
                with gr.Row():
                    pending_queue_elems = gr.Dropdown(
                        show_label=False,
                        visible=True,
                        interactive=True,
                    )
                    with gr.Column(scale=6):
                        pending_args = gr.DataFrame(visible=False, interactive=False)
                        pending_metrics = gr.DataFrame(visible=False, interactive=False)
                        cancel_btn = gr.Button(
                            "Cancel", visible=False, interactive=False
                        )
        with gr.Row():
            model_type = gr.Dropdown(
                label="Model Type",
                choices=["baseline", "finetuned", "custom"],
                interactive=True,
            )
            model = gr.Dropdown(
                label="Model",
                interactive=True,
                scale=2,
                choices=EVAL_MODELS["baseline"],
            )
            custom_model = gr.Textbox(
                label="Custom Model",
                visible=False,
                interactive=False,
                scale=2,
            )
            infer_backend = gr.Dropdown(
                label="Infer Backend",
                choices=["vllm", "vllm_engine"],
                interactive=True,
            )
            model_dir = gr.Textbox(
                label="Model Dir",
                value="/home/llm_models",
                interactive=True,
            )
        with gr.Row():
            dataset = gr.Dropdown(
                label="Dataset",
                choices=EVAL_DATASETS.keys(),
                multiselect=True,
                scale=2,
                interactive=True,
            )
            dataset_dir = gr.Textbox(
                label="Dataset Dir",
                value="/home/data/0_origin",
                interactive=True,
            )
            save_dir = gr.Textbox(
                label="Save Dir",
                value="/home/leap/evaluation/eval_result",
                interactive=True,
            )
            leaderboard_check = gr.Checkbox(label="Leaderboard", value=True)
        with gr.Row():
            device = gr.CheckboxGroup(
                choices=["0", "1", "2", "3"],
                label="Device",
                value=["0"],
            )
            tensor_parallel_size = gr.Dropdown(
                label="Tensor Parallel Size",
                choices=[1, 2, 4, 8, 16],
                value=1,
                interactive=True,
            )
            gpu_memory_utilization = gr.Slider(
                label="GPU Memory Utilization",
                minimum=0.05,
                maximum=1,
                value=0.9,
                step=0.05,
            )
        with gr.Row():

            @gr.render(
                inputs=[gpu_status], concurrency_id="render", concurrency_limit=1
            )
            def show_gpu_bars(gpu_status):
                try:
                    gpu_status = dict(gpu_status)
                    gpu_len = len(gpu_status["name"])
                    with gr.Column():
                        for idx in range(gpu_len):
                            name, memory_total, memory_used, load = (
                                gpu_status["name"][idx],
                                gpu_status["memory_total"][idx],
                                gpu_status["memory_used"][idx],
                                gpu_status["load"][idx],
                            )
                            gpu_bar_label = (
                                f"{idx} | {name} | {memory_used} / {memory_total} MB"
                            )
                            gpu_bar_value = load
                            gr.Slider(
                                label=gpu_bar_label,
                                value=gpu_bar_value,
                                visible=True,
                            )
                except:
                    pass

            with gr.Column(scale=2):
                with gr.Row():
                    max_model_length = gr.Slider(
                        label="Max Model Length",
                        minimum=4,
                        maximum=131072,
                        value=3072,
                        step=1,
                    )
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=1,
                        maximum=1024,
                        value=16,
                        step=1,
                    )
                    num_fewshot = gr.Slider(
                        label="Num Fewshot", minimum=-1, maximum=5, value=-1, step=1
                    )

                with gr.Row():
                    top_k = gr.Slider(
                        label="Top K", minimum=-1, maximum=50, value=-1, step=1
                    )
                    top_p = gr.Slider(
                        label="Top P", minimum=0.05, maximum=1, value=1, step=0.05
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.05,
                        maximum=1,
                        value=1,
                        step=0.05,
                    )

        with gr.Row():
            guide_btn = gr.Button("Usage Guide", elem_id="guide_btn")
            start_btn = gr.Button("START", elem_id="start_btn")

        with Modal(visible=False) as guide_modal:
            for key, guide in EVAL_USAGE_GUIDE.items():
                with gr.Tab(key):
                    gr.Markdown(guide)

        with gr.Row():
            with gr.Column(scale=1):
                current_task = gr.Textbox(
                    label="Current Task", scale=2, visible=True, interactive=False
                )
                running_args = gr.DataFrame(label="Running Args", visible=True, scale=1)
            with gr.Column(scale=2):
                progress_bar = gr.Slider(scale=4, visible=True, interactive=False)
                output_box = gr.Textbox(
                    label="Log", visible=True, interactive=False, scale=2, lines=10
                )

    start_btn.click(
        fn=request_listener.evaluate_request,
        inputs=[
            model_type,
            model,
            dataset,
            model_dir,
            dataset_dir,
            infer_backend,
            device,
            tensor_parallel_size,
            gpu_memory_utilization,
            max_model_length,
            max_new_tokens,
            num_fewshot,
            top_k,
            top_p,
            temperature,
            save_dir,
            custom_model,
            leaderboard_check,
        ],
        outputs=[],
    )
    guide_btn.click(
        fn=lambda: Modal(visible=True),
        inputs=[],
        outputs=[guide_modal],
    )
    cancel_btn.click(
        fn=request_listener.cancel_request,
        inputs=[pending_queue_elems],
        outputs=[pending_queue_elems],
    )
    gpu_status.change(
        fn=gpu_listener.update_gpu,
        inputs=[gpu_status],
        outputs=[device],
        show_progress=False,
    )
    model_type.change(
        fn=model_listener.update_eval_model_list,
        inputs=[model_type],
        outputs=[model, custom_model],
    )
    model.change(
        fn=model_listener.update_eval_model_len,
        inputs=[model],
        outputs=[max_model_length],
    )
    evaluate_tab.select(
        fn=queue_listener.update_dropdown,
        inputs=[queue],
        outputs=[finished_queue_elems, pending_queue_elems],
        show_progress=False,
    )
    finished_queue_elems.change(
        fn=queue_listener.select_queue,
        inputs=[finished_queue_elems],
        outputs=[finished_args, finished_metrics, cancel_btn],
    )
    pending_queue_elems.change(
        fn=queue_listener.select_queue,
        inputs=[pending_queue_elems],
        outputs=[pending_args, pending_metrics, cancel_btn],
    )
    return {
        "queue": queue,
        "current_task": current_task,
        "progress_bar": progress_bar,
        "output_box": output_box,
        "running_args": running_args,
        "gpu_status": gpu_status,
    }
