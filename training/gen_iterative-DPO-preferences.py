import argparse
import json
import torch
import os
import datetime

# Model Libs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def calc_logits_next_token(model : object, tokenizer : object, text : str) -> str:   
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    # We could use do_sample=False and disable top_k and top_p to get a deterministic output
    outputs = model.generate(**tokenized, max_new_tokens = 5, do_sample = False, temperature = 1.0, pad_token_id = tokenizer.eos_token_id)
    
    #TO:DO: Output Generation Logits here
    

    return None

def gen_DPO_preferences():
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths, including a merging flag
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistralai/Mistral-7B-Instruct-v0.2')

    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='Base-Model_Iter-1')

    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)

    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="models/")
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'training/iterative_training_queries/iteration-1_base-model_iterative-train-explanations_train.json')

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

    res = {}

    for q_id in queries:
        q_id_res = q_id.split("_")[0]
        if q_id_res not in res:
            res[q_id_res] = {"explanations_scores" : [], "ids" : []}

        res[q_id_res]["ids"].append(q_id)
        text = f'{queries[q_id]["text"].split("Answer:")[0]} Answer: '

        logits_next_token = calc_logits_next_token(model, tokenizer, text)
        res[q_id_res]["explanations_scores"].append((text, logits_next_token))

        with safe_open_w(f'{args.output_dir}DPO_Explanation-Preferences_{args.exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json') as output_file:
            output_file.write(json.dumps(res, ensure_ascii=False, indent=4))


def main():
    gen_DPO_preferences()
    # parse_DPO_preferences()

if __name__ == '__main__':
    main()