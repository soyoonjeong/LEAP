import requests
from .url import API_URL

get_eval_models = lambda: requests.get(API_URL + "/list/eval/model").json()
get_tuning_models = lambda: requests.get(API_URL + "/list/tuning/model").json()[
    "tuning"
]

MODEL_TYPE = {
    "Qwen/Qwen2.5-1.5B": "qwen_25",
    "Qwen/Qwen2.5-3B": "qwen_25",
    "Qwen/Qwen2.5-7B": "qwen_25",
    "Qwen/Qwen2.5-14B": "qwen_25",
    "meta-llama/Llama-3.1-8B": "llama_31",
    "google/gemma-1.1-2b-it": "gemma_11",
    "google/gemma-1.1-7b-it": "gemma_11",
    "google/gemma-2-2b": "gemma_2",
    "google/gemma-2-9b": "gemma_2",
}
# gemma2
GEMMA_2_CHAT_FORM = (
    "<bos><start_of_turn>user \n {instruction} \n <end_of_turn><start_of_turn>model"
)

# LLama 3.1 8B Instruction Chat Template
LLaMA_31_CHAT_FORM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|> {instruction} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# QWEN2.5 7B Instruction Chat Template
QWEN_25_CHAT_FORM = "<|im_start|> system\n {system_prompt} <|im_end|>\n<|im_start|> user\n {instruction} <|im_end|>\n<|im_start|>assistant"

CHAT_TEMPLATES = {
    "gemma_11": GEMMA_2_CHAT_FORM,
    "gemma_2": GEMMA_2_CHAT_FORM,
    "llama_31": LLaMA_31_CHAT_FORM,
    "qwen_25": QWEN_25_CHAT_FORM,
}
