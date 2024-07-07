import argparse
import json
import torch

# Local files
from .eval_prompt import output_prompt_res

# Model Libs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths, including a merging flag
    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='base-model_explain-query-10')

    # Model and checkpoint paths, including a merging flag
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="models/pre-train_run-1_MedMix/checkpoint-6526/")
    
    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)



    # Path to queries, qrels and prompt files
    parser.add_argument('--used_set', type=str, help='choose which data to use', default="train") # train | dev | test
    args = parser.parse_known_args()
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{args[0].used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{args[0].used_set}.json')

    parser.add_argument('--prompt_file', type=str, help='path to prompts file', default="prompts/DPOPrompts.json")
    parser.add_argument('--prompt_name', type=str, help='name of prompt to use', default="gen_explanations")

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    args = parser.parse_args()

    if args.merge:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, low_cpu_mem_usage=True,
            return_dict=True, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map= {"": 0})
        model = PeftModel.from_pretrained(
            model, args.checkpoint,
            return_dict=True, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
            device_map= {"": 0})
        model = model.merge_and_unload()

    else:
        model = AutoModelForCausalLM.from_pretrained(
                args.model, low_cpu_mem_usage=True,
                return_dict=True, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map= {"": 0})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompt = json.load(open(args.prompt_file))[args.prompt_name]

    output_prompt_res(model, tokenizer, queries, qrels, prompt, args, args.used_set)

if __name__ == '__main__':
    main()