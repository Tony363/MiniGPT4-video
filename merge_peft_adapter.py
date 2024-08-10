import torch
from typing import Optional
from peft import PeftConfig, PeftModel
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


"""
Merge LoRA adapters into language model:
python ha_dpo/models/minigpt4/merge_peft_adapter.py \
--adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/{model_name} \
--base_model_name meta-llama/Llama-2-7b-chat-hf \
--output_name {path_to_merged_llm}

1. --adapter_model_name: path to the saved adapter weights during training.
2. --output_name: path where the merged language model weights are saved.


python merge_peft_adapter.py \
--adapter_model_name minigpt4/training_output/engagenet/mistral_rppg_former/202408082024/checkpoint_best.pth \
--base_model_name mistralai/Mistral-7B-Instruct-v0.2 \
--output_name /home/tony/nvme2tb/tuned_models/adapted_mistral_rppg_former

python ha_dpo/models/minigpt4/merge_peft_adapter.py \
--adapter_model_name ha_dpo/models/minigpt4/minigpt4/output/minigpt4_dpo \
--base_model_name wangrongsheng/MiniGPT-4-LLaMA-7B \
--output_name ha_dpo/models/minigpt4/minigpt4/output/merged_minigpt4_hadpo   
"""
@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    


    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
#model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
