import argparse
import json
import torch

# Local files
import eval_prompt

# Model Libs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths, including a merging flag
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')

    # Path to queries, qrels and prompt files
    parser.add_argument('--used_set', type=str, help='choose which data to use', default="train") # train | dev | test
    args = parser.parse_known_args()
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/base_sets/queries2024_{args[0].used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/base_sets/qrels2024_{args[0].used_set}.json')
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/AddPrompts.json")

    # Task to run
    parser.add_argument('--task', type=str, help='task to run', default='output_labels', choices=['expand_text']) # expand_text

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
            args.model, low_cpu_mem_usage=True,
            return_dict=True, torch_dtype=torch.bfloat16,
            device_map= {"": 0}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompt = json.load(open(args.prompts))["expand_queries"]

    eval_prompt.output_prompt_res(model, tokenizer, queries, qrels, prompt, args, args.used_set)

if __name__ == '__main__':
    main()