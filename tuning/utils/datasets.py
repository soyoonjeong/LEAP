import copy
import torch
import transformers
from typing import Dict, Sequence
from transformers import AutoTokenizer
from tqdm import tqdm

from torch.utils.data import Dataset
from config.llm_option import IGNORE_INDEX
from config.chat_templates import LLaMA_31_CHAT_FORM, QWEN_25_CHAT_FORM


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer)
        for strings in tqdm((examples, sources), desc="Example, Sources Tokenizing")
    ]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in tqdm(
        zip(labels, sources_tokenized["input_ids_lens"]), desc="IGNORE_INDEX Convert"
    ):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        prompt_input: str,
        system_prompt: str,
        data_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super(SupervisedDataset, self).__init__()

        print("Data Len : " + str(len(data_dict)))
        print("Formatting inputs...")

        sources = []
        for example in data_dict:
            if example.get("instruction", "") != "":
                if system_prompt != "":
                    sources.append(
                        prompt_input.format(
                            system_prompt=system_prompt, instruction=example
                        )
                    )
                else:
                    sources.append(prompt_input.format(instruction=example))

        # NOTE : Add EOS Token
        targets = []
        for example in data_dict:
            targets.append(example["output"] + tokenizer.eos_token)

        print("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
