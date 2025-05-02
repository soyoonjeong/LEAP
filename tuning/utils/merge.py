import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoTokenizer


def adapter_merge(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    merged_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(args.merge_output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len
    tokenizer.save_pretrained(args.merge_output_dir)
