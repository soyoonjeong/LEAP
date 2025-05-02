# Gemma2
GEMMA_2_CHAT_FORM = (
    "<bos><start_of_turn>user \n {instruction} \n <end_of_turn><start_of_turn>model"
)

# LLama 3.1 8B Instruction Chat Template
LLaMA_31_CHAT_FORM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|> {instruction} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# QWEN2.5 7B Instruction Chat Template
QWEN_25_CHAT_FORM = "<|im_start|> system\n {system_prompt} <|im_end|>\n<|im_start|> user\n {instruction} <|im_end|>\n<|im_start|>assistant"
