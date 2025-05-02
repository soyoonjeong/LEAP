from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

import torch


@dataclass
class VllmArguments:
    """
    Arguments pertaining to the vLLM worker.
    """

    # vllm_maxlen: int = field(
    #     default=4096,
    #     metadata={
    #         "help": "Maximum sequence (prompt + response) length of the vLLM engine."
    #     },
    # )
    # vllm_gpu_util: float = field(
    #     default=0.95,
    #     metadata={
    #         "help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."
    #     },
    # )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )


@dataclass
class ModelArguments(VllmArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model: Optional[str] = field(  # 사용
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    model_dir: str = field(
        default="/home",
        metadata={"help": "Path to the folder containing models."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    use_accelerate: bool = field(  # 사용
        default=True,
        metadata={
            "help": "Whether or not to use accelerate to accelerate the evaluating process."
        },
    )
    model_revision: str = field(  # 사용할 수도 있음
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    infer_backend: Literal["huggingface", "vllm"] = field(  # 사용
        default="vllm",
        metadata={"help": "Backend engine used at inference."},
    )
    device: Optional[Union[int, str]] = field(
        default="cuda",
        metadata={"help": "Device to use for inference."},
    )
    device_map_option: Optional[str] = field(  # 사용
        default="auto",
        metadata={"help": "Device map options for the model. "},
    )
    max_memory_per_gpu: Optional[Union[int, str]] = field(  # 사용
        default=None,
        metadata={"help": "Max memory per GPU for the model."},
    )
    max_cpu_memory: Optional[Union[int, str]] = field(  # 사용
        default=None,
        metadata={"help": "Max CPU memory for the model."},
    )
    offload_folder: str = field(  # 사용
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(  # 사용
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    tensor_parallel_size: int = field(  # 사용
        default=2, metadata={"help": "Tensor Parallel size"}
    )
    max_model_len: int = field(
        default=4096,
        metadata={"help": "The maximum length the generated tokens can have."},
    )
    gpu_memory_utilization: float = field(
        default=0.95,
        metadata={"help": "The maximum memory utilization of GPU."},
    )
