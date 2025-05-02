# NOTE : 참고 자료 링크
# https://blog.eleuther.ai/transformer-math/
import json
import argparse
from typing import Literal, Union


class Training_Mem_Compute:
    @staticmethod
    def get_model_memory(
        model_params: Union[int, float],
        precision: Literal["fp16", "fp32", "bf16", "int8", "int4"],
    ) -> float:
        precision = precision.replace("bfloat16", "bf16")
        precision = precision.replace("float16", "fp16")
        precision = precision.replace("float32", "fp32")
        precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        model_memory_gb = model_params * precision_bytes[precision]

        return model_memory_gb

    @staticmethod
    def get_optimizer_memory(
        model_params: Union[int, float],
        optimizer: Literal["adamw", "bitsandbytes", "sgd"],
    ):
        optimizer_bytes = {"adamw": 12, "bitsandbytes": 6, "sgd": 8}
        optimizer_memory_gb = model_params * optimizer_bytes[optimizer]
        return optimizer_memory_gb

    @staticmethod
    def get_gradient_memory(
        model_params: Union[int, float],
        gradient: Literal["fp16", "fp32", "bf16", "int8", "int4"],
    ):
        gradient_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        gradient_memory_gb = model_params * gradient_bytes[gradient]
        return gradient_memory_gb

    @staticmethod
    def get_activation_memory(
        activation_checkpointing: Literal["none", "selective", "full"],
        batch_size: int,
        hidden_size: int,
        seq_length: int,
        num_layers: int,
        tensor_parallel_size: int,
        num_heads: int,
    ):
        if activation_checkpointing == "none":
            activation_memory = (
                batch_size
                * seq_length
                * hidden_size
                * num_layers
                * (
                    10
                    + 24 / tensor_parallel_size
                    + 5 * num_heads * seq_length / (hidden_size * tensor_parallel_size)
                )
                / (1024**3)
            )
        elif activation_checkpointing == "selective":
            activation_memory = (
                batch_size
                * seq_length
                * hidden_size
                * num_layers
                * (10 + 24 / tensor_parallel_size)
                / (1024**3)
            )
        elif activation_checkpointing == "full":
            activation_memory = (
                2 * batch_size * seq_length * hidden_size * num_layers / (1024**3)
            )
        else:
            raise ValueError(
                "Invalid activation_checkpointing. Choose 'none', 'selective', or 'full'."
            )
        return activation_memory


def get_model_info(model_name: str, num_params: float):

    with open(f"/home/{model_name}/config.json", "r") as f:
        config = json.load(f)

    return_dict = {
        "Model": model_name,
        "#Params(B)": num_params,
        "Precision": config["torch_dtype"],
        "Architecture": config["architectures"][0],
        "hidden_size": config["hidden_size"],
        "seq_length": config["max_position_embeddings"],
        "num_layers": (
            config["num_layers"]
            if "num_layers" in config.keys()
            else config["num_hidden_layers"]
        ),
        "num_heads": config["num_attention_heads"],
    }

    return return_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--gradient",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "bf16", "int8", "int4"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "bitsandbytes", "sgd"],
    )
    parser.add_argument(
        "--activation_checkpointing",
        type=str,
        default="full",
        choices=["none", "selective", "full"],
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    args = dict(args.__dict__)

    return args


def get_training_memory(
    model: str,
    num_params: float,
    gradient: str = "fp32",
    optimizer: str = "adamw",
    activation_checkpointing: str = "full",
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    model_info: dict = None,
):
    if model_info is None:
        model_info = get_model_info(model, num_params)

    model_mem = Training_Mem_Compute.get_model_memory(
        model_params=num_params, precision=model_info["Precision"]
    )
    optimizer_mem = Training_Mem_Compute.get_optimizer_memory(
        model_params=num_params, optimizer=optimizer
    )
    gradient_mem = Training_Mem_Compute.get_gradient_memory(
        model_params=num_params, gradient=gradient
    )
    activation_mem = Training_Mem_Compute.get_activation_memory(
        activation_checkpointing=activation_checkpointing,
        batch_size=batch_size,
        hidden_size=model_info["hidden_size"],
        seq_length=model_info["seq_length"],
        num_layers=model_info["num_layers"],
        tensor_parallel_size=tensor_parallel_size,
        num_heads=model_info["num_heads"],
    )

    return model_mem, optimizer_mem, gradient_mem, activation_mem


# DEBUG
if __name__ == "__main__":
    args = parse_args()
    model_info = get_model_info(args["model"])
    print(json.dumps(model_info, indent=2))

    model_mem = Training_Mem_Compute.get_model_memory(
        model_params=model_info["num_params"], precision=model_info["precision"]
    )
    optimizer_mem = Training_Mem_Compute.get_optimizer_memory(
        model_params=model_info["num_params"], optimizer=args["optimizer"]
    )
    gradient_mem = Training_Mem_Compute.get_gradient_memory(
        model_params=model_info["num_params"], gradient=args["gradient"]
    )
    activation_mem = Training_Mem_Compute.get_activation_memory(
        activation_checkpointing=args["activation_checkpointing"],
        batch_size=args["batch_size"],
        hidden_size=model_info["hidden_size"],
        seq_length=model_info["seq_length"],
        num_layers=model_info["num_layers"],
        tensor_parallel_size=args["tensor_parallel_size"],
        num_heads=model_info["num_heads"],
    )
    print(f"Model Memory: {model_mem}")
    print(f"Optimizer Memory: {optimizer_mem}")
    print(f"Gradient Memory: {gradient_mem}")
    print(f"Activation Memory: {activation_mem}")
    total_training_mem = model_mem + optimizer_mem + gradient_mem + activation_mem
    print("Total Training Memory: {0:.2f} GB".format(total_training_mem))
