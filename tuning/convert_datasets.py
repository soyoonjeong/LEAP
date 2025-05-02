import os
import argparse

from datasets import Dataset


from config.setting import ROOT_DIR
from config.support_models import SUPPORT_MODELS_TEMPLATES
from utils.file_utlis import read_json_file
from utils.datasets import SupervisedDataset
from utils.tokenizer import set_tokenizer


def make_datasets(args: argparse) -> None:
    # step 1 : Data Read
    data_list = read_json_file(args.instruction_data_path)
    # step 3 : Set Tokenizer
    tokenizer = set_tokenizer(args.model, args.max_seq_len)
    # step 4 : Dataset convert
    dataset = SupervisedDataset(
        SUPPORT_MODELS_TEMPLATES[args.model_type],
        args.system_prompt,
        data_dict=data_list,
        tokenizer=tokenizer,
    )
    dataset = Dataset.from_list(dataset)
    # step 5 : Dataset Save
    dataset.save_to_disk(args.result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="File Concat", description="-")
    args = parser.add_argument_group("Create multi document dataset")
    args.add_argument(
        "--system_prompt",
        type=str,
        default="-",
        help="User : system_prompt",
    )
    args.add_argument(
        "--max_seq_len",
        type=int,
        default=3072,
        help="max_seq_len",
    )
    args.add_argument(
        "--instruction_data_path",
        type=str,
        default=os.path.join(
            ROOT_DIR,
            "ex_data/user_example_data.json",
        ),
        help="파일 경로",
    )
    args.add_argument(
        "--model",
        type=str,
        default=os.path.join(ROOT_DIR, "gemma2-2b-it"),
        help="model",
    )
    args.add_argument(
        "--model_type",
        type=str,
        default="gemma_2",
        help="Model type : ex qwen_25, llama_31, gemma_2",
    )
    args.add_argument(
        "--result_path",
        type=str,
        default=os.path.join(ROOT_DIR, "gemma2-it-1k-Test-Dataset"),
        help="저장 경로",
    )

    main_args = parser.parse_args()
    make_datasets(main_args)
