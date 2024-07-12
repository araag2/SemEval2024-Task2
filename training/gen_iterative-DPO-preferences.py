import argparse
import json
import torch
import os
import re



# Model Libs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def calc_logits_next_token(model : object, tokenizer : object, text : str, token_lists : dict) -> str:   
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    # We could use do_sample=False and disable top_k and top_p to get a deterministic output
    outputs = model.generate(**tokenized, max_new_tokens = 5, do_sample = False, temperature = 1.0, pad_token_id = tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)

    greedy_score = outputs.scores[0]

    no_scores = [greedy_score[0][token[-1]] for token in token_lists["no_tokens"]]
    yes_scores = [greedy_score[0][token[-1]] for token in token_lists["yes_tokens"]]

    max_scores = torch.softmax(torch.tensor([max(no_scores), max(yes_scores)]), dim=0).tolist()
    avg_scores = torch.softmax(torch.tensor([sum(no_scores)/3.0, sum(yes_scores)/3.0]), dim=0).tolist()

    return {"max_scores" : max_scores, "avg_scores" : avg_scores}

def order_DPO_preferences(res):
    for q_id_res in res:
        res[q_id_res]["ids"] = [id for _, id in sorted(zip(res[q_id_res]["explanations_scores"], res[q_id_res]["ids"]), key=lambda x: x[0][1], reverse=True)]
        res[q_id_res]["explanations_scores"] = sorted(res[q_id_res]["explanations_scores"], key=lambda x: x[1], reverse=True)

    prompt = json.load(open("prompts/DPOPrompts.json"))["gen_explanations"]

    dpo_preferences_train = {"prompt" : [], "chosen" : [], "rejected" : []}
    dpo_preferences_dev = {"prompt" : [], "chosen" : [], "rejected" : []}

    n_train = 0

    for q_id_res in res:
        if len(res[q_id_res]["explanations_scores"]) >= 7:

            for i in range(3):
                dpo_preferences_train["prompt"].append(prompt)
                dpo_preferences_train["chosen"].append(res[q_id_res]["explanations_scores"][i][0])
                dpo_preferences_train["rejected"].append(res[q_id_res]["explanations_scores"][-(i+1)][0])
                n_train += 1

            # 1/9 of the data is used for dev
            if n_train % 9 == 0: 
                dpo_preferences_dev["prompt"].append(prompt)
                dpo_preferences_dev["chosen"].append(res[q_id_res]["explanations_scores"][3][0])
                dpo_preferences_dev["rejected"].append(res[q_id_res]["explanations_scores"][-4][0])
    
    return dpo_preferences_train, dpo_preferences_dev
    

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

    token_lists = {"no_tokens" : [tokenizer.encode("N"), tokenizer.encode("No"), tokenizer.encode("no")], "yes_tokens" : [tokenizer.encode("Y"), tokenizer.encode("Yes"), tokenizer.encode("yes")]}

    for q_id in tqdm(queries):
        q_id_res = q_id.split("_")[0]

        text = f'{queries[q_id]["text"].split("Answer:")[0]} Answer:'
        explanation = re.search(".*The following is an explanation of the statement's validity, by another expert using on the information available within the CTRs.\\n\\n(.*)\\n\\nBased on your analysis of the CTRs,.*", text)

        if explanation != None:

            if q_id_res not in res:
                res[q_id_res] = {"explanations_scores" : [], "ids" : [], "gold_label" : 0 if "No" in queries[q_id]["text"].split("Answer:")[1] else 1}

            res[q_id_res]["ids"].append(q_id)
            logits_next_token = calc_logits_next_token(model, tokenizer, text, token_lists)

            if explanation != None:
                res[q_id_res]["explanations_scores"].append((explanation.group(1), logits_next_token["max_scores"][res[q_id_res]["gold_label"]]))

    res_train, res_dev = order_DPO_preferences(res)

    with safe_open_w(f'{args.output_dir}DPO_Explanation-Scores_{args.exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json') as output_file:
        output_file.write(json.dumps(res, ensure_ascii=False, indent=4))

    with safe_open_w(f'{args.output_dir}DPO_Preferences-Train_{args.exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json') as output_file:
        output_file.write(json.dumps(res_train, ensure_ascii=False, indent=4))

    with safe_open_w(f'{args.output_dir}DPO_Preferences-Dev_{args.exp_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.json') as output_file:
        output_file.write(json.dumps(res_dev, ensure_ascii=False, indent=4))

def main():
    gen_DPO_preferences()

if __name__ == '__main__':
    main()