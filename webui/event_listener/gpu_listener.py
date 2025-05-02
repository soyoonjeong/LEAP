import gradio as gr


def update_gpu(gpu_status):
    gpus = len(gpu_status["name"])
    device = gr.CheckboxGroup(
        choices=[str(idx) for idx in range(gpus)],
    )
    return device
