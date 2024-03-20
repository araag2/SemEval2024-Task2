import os
import argparse
import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--checkpoint', type=str, help='path to prompts file', default="../models/Mistral-7B-Instruct-v0.2/run_14/end_model/")
    args = parser.parse_args()

    base_model_reload = AutoModelForCausalLM.from_pretrained(
       args.model, low_cpu_mem_usage=True,
       return_dict=True,torch_dtype=torch.bfloat16,
       device_map= {"": 0}
    )

    model = PeftModel.from_pretrained(base_model_reload, args.checkpoint)
    model = model.merge_and_unload()
    model.save_pretrained(f'{args.checkpoint}/compiled/')

if __name__ == '__main__':
    main()