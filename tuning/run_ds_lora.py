import os
import copy
import torch

from typing import Optional, cast
from datasets import load_from_disk
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser, TrainingArguments, Trainer

from utils.peft_utils import SaveDeepSpeedPeftModelCallback, create_and_prepare_model
from utils.file_utlis import save_json
from config.setting import ROOT_DIR


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    Additional arguments for training, which are not part of TrainingArguments.
    """

    model: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "The preference dataset to use."},
    )
    tokenizer_model_max_length: int = field(
        metadata={"help": "Model Max Length"},
    )
    lora_r: Optional[int] = field(default=0)  # 64
    lora_alpha: Optional[int] = field(default=0)  # 16
    lora_dropout: Optional[float] = field(default=0.05)

    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )
    use_dora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables DoRA "},
    )
    use_rslora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables rsLoRA "},
    )


def training_function(script_args: ScriptArguments, training_args: TrainingArguments):
    # Load processed dataset from disk
    dataset = load_from_disk(script_args.dataset_path)
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(
        script_args.model, training_args, script_args
    )
    model.config.use_cache = False

    # Create trainer and add callbacks
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    if os.path.isdir(training_args.output_dir) == False:
        os.makedirs(training_args.output_dir, exist_ok=True)
        # SAVE Tuning Parameters

    script_args_dict = copy.deepcopy(vars(script_args))
    training_args_dict = copy.deepcopy(asdict(training_args))
    combined_args_dict = {**script_args_dict, **training_args_dict}
    save_json(
        combined_args_dict,
        training_args.output_dir + "/tuning_parameters.json",
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()
    trainer.add_callback(
        SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps)
    )

    # Start training
    trainer.train()

    # Save model on main process
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    trainer.accelerator.wait_for_everyone()

    # TODO: add merge adapters
    # Save everything else on main process
    if trainer.args.process_index == 0:
        if script_args.merge_adapters:
            # merge adapter weights with base model and save
            # save int 4 model
            trainer.model.save_pretrained(
                training_args.output_dir, safe_serialization=False
            )
            # clear memory
            del model
            del trainer
            torch.cuda.empty_cache()

            from peft import AutoPeftModelForCausalLM

            # load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            # Merge LoRA and base model and save
            model = model.merge_and_unload()
            model.save_pretrained(
                training_args.output_dir + "/Merge/",
                safe_serialization=True,
                max_shard_size="8GB",
            )
        else:
            trainer.model.save_pretrained(
                training_args.output_dir + "/Final/", safe_serialization=True
            )
            # save tokenizer
        tokenizer.save_pretrained(training_args.output_dir + "/Tokenizer/")


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
