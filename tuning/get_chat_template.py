from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "/home/google/gemma-1.1-2b-it"
dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_id)
chat = [
    {"role": "user", "content": "Write a hello world program"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)
