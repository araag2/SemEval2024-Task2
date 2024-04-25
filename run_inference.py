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
    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='run-2_self-consistency_top-5_4')

    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)

    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="models/run_2_self-consistency_manual-expand/checkpoint-2930/")

    parser.add_argument('--constraint', dest='constraint', action='store_true', help='boolean flag to set if model is constrained on Yes or No')
    parser.add_argument('--no-constraint', dest='constraint', action='store_true', help='boolean flag to set if model is constrained on Yes or No')
    parser.set_defaults(constraint=False)


    # Path to queries, qrels and prompt files
    parser.add_argument('--used_set', type=str, help='choose which data to use', default="test_train-self-consistency_top-5") # train | dev | test
    args = parser.parse_known_args()
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{args[0].used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{args[0].used_set}.json')
    
    parser.add_argument('--prompt_file', type=str, help='path to prompts file', default="prompts/AddPrompts.json")
    parser.add_argument('--prompt_name', type=str, help='name of the prompt to use', default='explain_entailment_or_contradiction_prompt')

    # Task to run
    parser.add_argument('--task', type=str, help='task to run', default='output_labels', choices=['output_labels', 'evaluate']) # output_labels | self_consistency | evaluate

    parser.add_argument('--task_type', type=str, help='task type to run', default='self_consistency', choices=['base', 'self_consistency', 'explain_answer']) # output_labels | self_consistency | evaluate

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    args = parser.parse_args()

    model = None

    if args.merge:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        model = PeftModel.from_pretrained(model, args.checkpoint, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        model = model.merge_and_unload()
    else:
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
    prompt = json.load(open(args.prompt_file))[args.prompt_name]

    if args.task == "output_labels":
        eval_prompt.output_prompt_labels(model, tokenizer, queries, prompt, args, args.used_set, args.constraint, args.task_type)

    elif args.task == "evaluate":
        eval_prompt.full_evaluate_prompt(model, tokenizer, queries, qrels, "id-best_combination_prompt", prompt, args, args.used_set, args.task_type)

if __name__ == '__main__':
    main()