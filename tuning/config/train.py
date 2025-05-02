PER_STEP_CMD = {
    "convert_datasets": ["python", "tuning/convert_datasets.py"],
    "run_tuning": ["torchrun", "--nproc_per_node", "1", "tuning/run_ds_lora.py"],
    "run_merge": ["python", "tuning/run_merge.py"],
}
PER_STEP_ARG = {
    "convert_datasets": [
        "system_prompt",
        "max_seq_len",
        "instruction_data_path",
        "model",
        "model_type",  # chat template 맞춰주기 위한 것
        "result_path",
    ],
    "run_tuning": [
        "model",
        "dataset_path",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "use_flash_attn",
        "merge_adapters",
        "use_dora",
        "use_rslora",
        "output_dir",
        "save_strategy",
        "num_train_epochs",
        "logging_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_grad_norm",
        "weight_decay",
        "lr_scheduler_type",
        "gradient_checkpointing",
        "bf16",
        "fp16",
        "deepspeed",
        "warmup_ratio",
        "tokenizer_model_max_length",
    ],
    "run_merge": [
        "model",
        "adapter_path",
        "max_seq_len",
        "merge_output_dir",
    ],
}

train_input = {
    "system_prompt": "",
    "max_seq_len": 1024,  # convert_dataset: max_seq_len, tuning: tokenizer_model_max_length
    "instruction_data_path": "ex_data/user_example_data.json",
    "model_path": "/home/llm_models/google/gemma-2-2b-it",
    "model_type": "gemma-2",
    "result_path": "gemma-2-2b-it-1k-Test-Dataset",
    "model_id": "gemma-2-2b-it",
    "dataset_path": "ex_data/test_dataset.json",  # [사용자 입력 x], convert_dataset하고 그 경로 그대로 사용
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "use_flash_attn": False,
    "merge_adapters": False,
    "use_dora": False,
    "use_qlora": False,
    "output_dir": "gemma2-2b-it-LoRA",  # [사용자 입력 x]
    "save_strategy": "steps",  # [사용자 입력 x]
    "num_train_epochs": 5.0,
    "logging_steps": 1,  # [사용자 입력 x]
    "save_steps": 2,
    "save_total_limit": 200,  # [사용자 입력 x]
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 5,
    "learning_rate": 1e-4,
    "max_grad_norm": 1.0,
    "weight_decay": 0.001,
    "lr_scheduler_type": "linear",
    "gradient_checkpointing": True,  # [사용자 입력 x]
    "bf16": True,  # 모델 정밀도에 따라 결정
    "fp16": False,
    "deepspeed": "ds_stage2.json",
    "optimizer": "Adamw",  # [사용자 입력 x]
    "warmup_ratio": 0,
}
