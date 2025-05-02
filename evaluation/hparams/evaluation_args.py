import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to specify the evaluation parameters.
    """

    task: str = field(
        metadata={"help": "Name of the evaluation task."},
    )
    task_dir: str = field(
        default="/home/data/0_origin",
        metadata={"help": "Path to the folder containing the evaluation datasets."},
    )
    batch_size: str = field(
        default="auto",
        metadata={"help": "The batch size per GPU for evaluation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed to be used with data loaders."},
    )
    lang: Literal["en", "ko"] = field(
        default="ko",
        metadata={"help": "Language used at evaluation."},
    )
    num_fewshot: int = field(
        default=-1,
        metadata={"help": "Number of examplars for few-shot learning."},
    )
    save_dir: Optional[str] = field(
        default="/home/leap/eval_result/",
        metadata={"help": "Path to save the evaluation results."},
    )
    max_concurrent_tasks: int = field(
        default=64,
        metadata={
            "help": "Limit the maximum number of simultaneous requests to avoid OOM(OutOfMemory) problem"
        },
    )
    cutoff_len: int = field(
        default=1024,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    max_samples: Optional[int] = field(
        default=100000,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."
        },
    )

    def __post_init__(self):
        # task 여러 개 입력받는 경우 리스트로 처리
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.task = split_arg(self.task)
