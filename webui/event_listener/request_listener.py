import os
import json
import requests
import gradio as gr

from config.url import API_URL
from config.path import SAVE_MODEL_DIR, DATASET_DIR, SAVE_DATASET_DIR


# START 버튼 누르면 실행
def evaluate_request(
    model_type: str,
    model: str,
    dataset: list,
    model_dir: str,
    dataset_dir: str,
    infer_backend: str,
    device: list,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_length: int,
    max_new_tokens: int,
    num_fewshot: int,
    top_k: int,
    top_p: float,
    temperature: float,
    save_dir: str,
    custom_model: str,
    leaderboard_check: bool,
):
    args = {
        "model": custom_model if model_type == "custom" else model,
        "task": dataset,
        "infer_backend": infer_backend,
        "model_dir": model_dir,
        "task_dir": dataset_dir,
        "num_fewshot": num_fewshot,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_length,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "device": device,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "save_dir": save_dir,
        "write_out": leaderboard_check,
    }

    error_flag = False
    if len(dataset) == 0:
        gr.Warning("Dataset을 선택해주세요.")
        error_flag = True
    if len(device) == 0:
        gr.Warning("Device를 선택해주세요.")
        error_flag = True
    if len(device) < tensor_parallel_size:
        gr.Warning(f"Tensor Parallel Size를 {len(device)} 이하 수로 설정해주세요.")
        error_flag = True

    try:
        if not error_flag:
            response = requests.post(
                API_URL + "/evaluate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(args),
            )
            gr.Info("Evaluate 요청이 접수되었습니다.")
    except:
        pass


# STOP 버튼 누르면 실행
def stop_request():
    try:
        response = requests.post(
            API_URL + "/stop",
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(e)


# CANCEL 버튼 누르면 실행
def cancel_request(queue_elems):
    try:
        response = requests.post(
            API_URL + f"/cancel/{queue_elems['id']}",
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        print(e)

    pending_queue_elems = gr.Radio(
        label="Tasks", visible=True, interactive=True, elem_id="radio", value=None
    )
    return pending_queue_elems


def tuning_request(
    device: list,
    model_path: str,
    model_type: str,
    instruction_data_path: str,
    converted_data_path: str,
    system_prompt: str,
    learning_rate: str,
    lr_scheduler_type: str,
    warmup_ratio: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_seq_len: int,
    max_grad_norm: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    compute_type: str,
    deepspeed: str,
    use_flash_attn: bool,
    use_peft: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    merge_adapters: bool,
    output_dir: str,
    logging_steps: int,
    save_strategy: str,
):
    # 모든 로컬 변수를 가져옵니다.
    args = locals()

    # 추가로 필요한 값이나 변환이 필요한 항목 처리
    args.update(
        {
            "bf16": compute_type == "bf16",
            "fp16": compute_type == "fp16",
            "tokenizer_model_max_length": max_seq_len,
            "dataset_path": converted_data_path,
            "result_path": converted_data_path,
            "model": model_path,
            "use_dora": use_peft == "use_dora",
            "use_rslora": use_peft == "use_rslora",
        }
    )
    if instruction_data_path.startswith(str(SAVE_DATASET_DIR).split("/")[-1]):
        args["instruction_data_path"] = os.path.join(DATASET_DIR, instruction_data_path)
    else:
        args["instruction_data_path"] = os.path.join(
            DATASET_DIR,
            "Runpod-LoRA-Tuning-Datasets-max-seq-len-3k",
            "1_instruction_dataset",
            instruction_data_path,
        )
    args["output_dir"] = os.path.join(SAVE_MODEL_DIR, output_dir)
    args["deepspeed"] = f"tuning/{deepspeed}"

    error_flag = False
    if len(device) == 0:
        gr.Warning(f"Device를 1개 이상 선택해주세요.")
        error_flag = True

    try:
        if not error_flag:
            response = requests.post(
                API_URL + "/tuning",
                headers={"Content-Type": "application/json"},
                data=json.dumps(args),
            )
            gr.Info("Tuning 요청이 접수되었습니다.")
    except:
        pass
