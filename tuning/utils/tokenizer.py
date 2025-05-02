from transformers import AutoTokenizer


def set_tokenizer(path: str, max_seq_len: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.model_max_length = max_seq_len
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
