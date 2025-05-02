import gradio as gr

from gradio_modal import Modal

from event_listener import (
    request_listener,
    queue_listener,
    model_listener,
    gpu_listener,
)

from config import get_tuning_datasets
from config.guide import TUNING_USAGE_GUIDE


def create_tuning_tab(TUNING_MODELS, TUNING_DATASETS):
    with gr.Tab("🧠 Tuning") as tuning_tab:
        queue = gr.JSON(label="Queue", visible=False)
        gpu_status = gr.JSON(label="GPU Status", visible=False)
        with gr.Accordion(
            "⏳ Pending Tuning Queue",
            open=False,
        ):
            with gr.Row():
                pending_queue_elems = gr.Dropdown(
                    show_label=False,
                    visible=True,
                    interactive=True,
                    elem_id="radio",
                )
                with gr.Column(scale=6):
                    pending_args = gr.DataFrame(visible=False, interactive=False)
                    pending_metrics = gr.DataFrame(visible=False, interactive=False)
                    cancel_btn = gr.Button("Cancel", visible=False, interactive=False)
        with gr.Row():
            with gr.Column():
                device = gr.CheckboxGroup(
                    choices=["0", "1", "2", "3"],
                    label="Device",
                    value=["0"],
                )

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
                                gpu_bar_label = f"{idx} | {name} | {memory_used} / {memory_total} MB"
                                gpu_bar_value = load
                                gr.Slider(
                                    label=gpu_bar_label,
                                    value=gpu_bar_value,
                                    visible=True,
                                    scale=2,
                                    interactive=False,
                                )
                    except:
                        pass

            with gr.Column(scale=2):
                with gr.Row():
                    model = gr.Dropdown(
                        label="Model",
                        choices=TUNING_MODELS,
                        interactive=True,
                    )
                    model_path = gr.Textbox(
                        label="Model Path",
                        value="/home/llm_models/Qwen/Qwen2.5-1.5B",
                        interactive=True,
                    )
                with gr.Row():
                    instruction_data_path = gr.Dropdown(
                        TUNING_DATASETS,
                        label="Instruction Data Path",
                        visible=True,
                        interactive=True,
                        value="llm-tuning-dataset/user_example_data.json",
                    )
                    converted_data_path = gr.Textbox(
                        label="Converted Data Path",
                        value="dataset/Qwen2.5-1.5B-Dataset",
                        interactive=True,
                        visible=False,
                    )
                    output_dir = gr.Textbox(
                        label="Tuning Model Name",
                        value="Qwen2.5-1.5B-LoRA",
                        interactive=True,
                        visible=True,
                    )
                with gr.Row():
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value=" ",
                        interactive=True,
                    )
            with gr.Column():
                model_type = gr.Textbox(
                    label="Model Type",
                    value="qwen_25",
                    visible=True,
                    interactive=False,
                )
                chat_template = gr.Textbox(
                    label="Chat Template",
                    value="<|im_start|> system\n {system_prompt} <|im_end|>\n<|im_start|> user\n {instruction} <|im_end|>\n<|im_start|>assistant",
                    interactive=True,
                    show_copy_button=True,
                    lines=8,
                )

        with gr.Accordion("Optimization Configurations", open=False):
            with gr.Row():
                optimizer = gr.Textbox(
                    label="Optimizer",
                    value="adamw_torch",
                    interactive=False,
                )
                learning_rate = gr.Textbox(
                    label="Learning Rate",
                    value="1e-4",
                    interactive=True,
                )
                lr_scheduler_type = gr.Dropdown(
                    label="lr Scheduler",
                    choices=[
                        "linear",
                        "cosine",
                        "constant",
                        "constant_with_warmup",
                    ],
                    value="linear",
                    interactive=True,
                )
                warmup_ratio = gr.Number(
                    label="Warmup Ratio",
                    value=0,
                    step=0.01,
                    interactive=True,
                )
                weight_decay = gr.Number(
                    label="Weight Decay",
                    value=0.001,
                    step=0.001,
                    interactive=True,
                )
                gradient_accumulation_steps = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=5,
                    step=1,
                    interactive=True,
                )
                gradient_checkpointing = gr.Checkbox(
                    label="Gradient Checkpointing",
                    value=True,
                    interactive=False,
                )

        with gr.Accordion("Tuning Configurations", open=False):
            with gr.Row():
                max_seq_len = gr.Number(
                    label="Max Sequence Length",
                    value=1024,
                    step=1024,
                    maximum=9216,
                    interactive=True,
                )
                max_grad_norm = gr.Number(
                    label="Max Gradient Norm",
                    minimum=0,
                    maximum=10,
                    value=1,
                    step=0.1,
                    interactive=True,
                )
                num_train_epochs = gr.Number(
                    label="Epochs",
                    value=5,
                    interactive=True,
                )
                per_device_train_batch_size = gr.Number(
                    label="Per Device Batch Size",
                    value=1,
                    interactive=True,
                )
                compute_type = gr.Dropdown(
                    label="Compute Type",
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    interactive=True,
                )
                deepspeed = gr.Dropdown(
                    label="Deepspeed Config",
                    choices=["ds_stage2.json", "ds_stage3.json"],
                    value="ds_stage2.json",
                    interactive=True,
                )
                use_flash_attn = gr.Checkbox(
                    label="Use Flash Attention",
                    value=False,
                    interactive=True,
                )

        with gr.Accordion("LoRA Configurations", open=False):
            with gr.Row():
                use_peft = gr.Radio(
                    choices=["use_lora", "use_dora", "use_rslora"],
                    value="use_lora",
                    interactive=True,
                    show_label=False,
                    scale=2,
                )
                lora_r = gr.Number(
                    label="LoRA r",
                    minimum=1,
                    maximum=128,
                    value=8,
                    step=1,
                    interactive=True,
                )
                lora_alpha = gr.Number(
                    label="LoRA alpha",
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    interactive=True,
                )
                lora_dropout = gr.Number(
                    label="LoRA dropout",
                    minimum=0,
                    maximum=0.3,
                    value=0.1,
                    step=0.01,
                    interactive=True,
                )
                logging_steps = gr.Number(
                    label="Logging Steps", value=1, interactive=False, visible=False
                )
                save_strategy = gr.Textbox(
                    label="Save Strategy",
                    value="epoch",
                    interactive=False,
                    visible=True,
                )
                merge_adapters = gr.Checkbox(
                    label="Merge Adapters", value=True, interactive=True
                )

        with gr.Row():
            guide_btn = gr.Button("Usage Guide", elem_id="guide_btn")
            start_btn = gr.Button("START", elem_id="start_btn")

        with Modal(visible=False) as guide_modal:
            for key, guide in TUNING_USAGE_GUIDE.items():
                with gr.Tab(key):
                    gr.Markdown(guide)

        with gr.Row():
            with gr.Column(scale=1):
                current_task = gr.Textbox(
                    label="Current Task", visible=True, interactive=False
                )
                with gr.Accordion("Epoch vs Loss Graph", open=True):
                    loss_graph = gr.Plot(label="Loss", visible=True)
                with gr.Accordion("Epoch vs Grad Norm Graph", open=False):
                    grad_norm_graph = gr.Plot(label="Grad Norm", visible=True)
                running_args = gr.DataFrame(label="Running Args", visible=True, scale=1)
            with gr.Column(scale=2):
                progress_bar = gr.Slider(scale=4, visible=True, interactive=False)
                output_box = gr.Textbox(
                    label="Log", visible=True, interactive=False, scale=2, lines=12
                )
    start_btn.click(
        fn=request_listener.tuning_request,
        inputs=[
            device,
            model_path,
            model_type,
            instruction_data_path,
            converted_data_path,
            system_prompt,
            learning_rate,
            lr_scheduler_type,
            warmup_ratio,
            weight_decay,
            gradient_accumulation_steps,
            gradient_checkpointing,
            max_seq_len,
            max_grad_norm,
            num_train_epochs,
            per_device_train_batch_size,
            compute_type,
            deepspeed,
            use_flash_attn,
            use_peft,
            lora_r,
            lora_alpha,
            lora_dropout,
            merge_adapters,
            output_dir,
            logging_steps,
            save_strategy,
        ],
        outputs=[],
    )
    cancel_btn.click(
        fn=request_listener.cancel_request,
        inputs=[pending_queue_elems],
        outputs=[pending_queue_elems],
    )
    guide_btn.click(
        fn=lambda: Modal(visible=True),
        inputs=[],
        outputs=[guide_modal],
    )
    gpu_status.change(
        fn=gpu_listener.update_gpu,
        inputs=[gpu_status],
        outputs=[device],
        show_progress=False,
    )
    pending_queue_elems.change(
        fn=queue_listener.select_queue,
        inputs=[pending_queue_elems],
        outputs=[pending_args, pending_metrics, cancel_btn],
    )
    model.change(
        fn=model_listener.update_tuning_model,
        inputs=[model],
        outputs=[
            model_path,
            model_type,
            chat_template,
            converted_data_path,
            output_dir,
        ],
    )
    use_peft.change(
        fn=model_listener.update_peft,
        inputs=[use_peft],
        outputs=[lora_alpha],
    )
    tuning_tab.select(
        fn=lambda: gr.Dropdown(choices=get_tuning_datasets()),
        inputs=[],
        outputs=[instruction_data_path],
    )
    return {
        "queue": queue,
        "current_task": current_task,
        "loss_graph": loss_graph,
        "grad_norm_graph": grad_norm_graph,
        "running_args": running_args,
        "progress_bar": progress_bar,
        "output_box": output_box,
        "gpu_status": gpu_status,
        "pending_queue_elems": pending_queue_elems,
        "cancel_btn": cancel_btn,
        "tuning_tab": tuning_tab,
    }
