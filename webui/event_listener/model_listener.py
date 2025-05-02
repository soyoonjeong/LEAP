import os
import tempfile
import gradio as gr
from transformers import PretrainedConfig

from config.model import MODEL_TYPE, CHAT_TEMPLATES, get_eval_models
from config.path import MODEL_DIR


def update_eval_model_list(model_type):
    EVAL_MODELS = get_eval_models()
    custom_model = gr.Textbox(label="Custom Model", visible=False, interactive=False)
    if model_type == "finetuned":
        model = gr.Dropdown(
            label="Model", choices=EVAL_MODELS["finetuned"], visible=True
        )
    elif model_type == "baseline":
        model = gr.Dropdown(
            label="Model", choices=EVAL_MODELS["baseline"], visible=True
        )
    else:
        model = gr.Dropdown(label="Model", visible=False)
        custom_model = gr.Textbox(label="Custom Model", visible=True, interactive=True)

    return model, custom_model


def update_eval_model_len(model):
    # max_model_length
    try:
        #
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PretrainedConfig.from_pretrained(model, cache_dir=tmpdir)
        max_model_length = config.max_position_embeddings
    except:
        max_model_length = 4096

    return gr.Slider(
        label="Max Model Length",
        minimum=4,
        maximum=max_model_length,
        value=3072,
        step=1,
    )


def update_tuning_model(model):
    model_path = os.path.join(MODEL_DIR, model)
    model_type = MODEL_TYPE[model]
    chat_template = CHAT_TEMPLATES[model_type]
    converted_data_path = f"dataset/{model.split('/')[-1]}-Dataset"
    output_dir = f"{model.split('/')[-1]}-LoRA"

    return (
        model_path,
        model_type,
        chat_template,
        converted_data_path,
        output_dir,
    )


def update_peft(radio):
    if radio == "use_lora":
        return gr.Number(visible=True)
    else:
        return gr.Number(visible=False)
