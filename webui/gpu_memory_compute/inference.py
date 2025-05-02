# NOTE : 참고 자료 링크
# https://blog.eleuther.ai/transformer-math/
import sys
import json
import argparse
from typing import Literal, Union

from config.model import get_eval_models


class Inference_Mem_Compute:
    @staticmethod
    def get_inference_memory(
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
    def get_overhead_memory(inference_memory: float) -> float:
        return inference_memory * 1.2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=get_eval_models()["baseline"])
    args = parser.parse_args()
    args = dict(args.__dict__)
    return args


def get_model_info(model_name: str, num_params: float):

    with open(f"/home/{model_name}/config.json", "r") as f:
        config = json.load(f)

    return_dict = {
        "Model": model_name,
        "#Params(B)": num_params,
        "Precision": config["torch_dtype"],
    }

    return return_dict


def get_inference_memory(num_params: float):
    if_mem = Inference_Mem_Compute.get_inference_memory(
        model_params=num_params,
        precision="float16",  # torch_dtype
    )
    over_head = Inference_Mem_Compute.get_overhead_memory(if_mem)

    return over_head


# DEBUG
if __name__ == "__main__":
    args = parse_args()
    model_info = get_model_info(args["model"])
    print(model_info)

    if_mem = Inference_Mem_Compute.get_inference_memory(
        model_params=model_info["num_params"],
        precision=model_info["precision"],  # torch_dtype
    )
    over_head = Inference_Mem_Compute.get_overhead_memory(if_mem)

    print("Total Inference Memory: {0:.2f} GB ~ {1:.2f} GB".format(if_mem, over_head))
