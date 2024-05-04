import os
import json
import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def tokenize_generate_decode(model : object, tokenizer : object, text : str, max_new_tokens : int = 50, top_k : int = 50, top_p : float = 0.95, do_sample : bool = True, temperature : float = 1.0) -> str:   
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    # We could use do_sample=False and disable top_k and top_p to get a deterministic output
    outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens, top_k = top_k, top_p = top_p, do_sample=do_sample, temperature = temperature, pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--queries_2023', type=str, default='../queries/queries2023.json')
    parser.add_argument('--queries_2024_test', type=str, default='../queries/queries2024_train.json')
    parser.add_argument('--queries_2024_dev', type=str, default='../queries/queries2024_dev.json')
    parser.add_argument('--queries_2024_self-consistency', type=str, default='../queries/task_info/queries2024_train_self-consistency_manual-expand_and_dev.json')
    parser.add_argument('--output_name', type=str, default='../qrels/qrels2024_normal_self-consistency_evidence-index.json')
    args = parser.parse_args()

    q_2023 = json.load(open(args.queries_2023)) 
    q_test_2024 = json.load(open(args.queries_2024_test))
    q_dev_2024 = json.load(open(args.queries_2024_dev))
    q_self_consistency_2024 = json.load(open(args.queries_2024_self_consistency))


    res = {}

    #for q_id in tqdm(q_2023):
    #    res[f'{q_id}'] = q_test_2024[q_id] if q_id in q_test_2024 else q_dev_2024[q_id]
    #
    #    res[f'{q_id}_evidence'] = res[f'{q_id}']
    #    res[f'{q_id}_evidence']['primary_evidence'] = "".join([res[f'{q_id}']["Primary_id_txt_list"][i] for i in q_2023[q_id]["Primary_evidence_index"]])
    #    if "Secondary_evidence_index" in q_2023[q_id]:
    #        res[f'{q_id}_evidence']['secondary_evidence'] = "".join([res[f'{q_id}']["Secondary_id_txt_list"][i] for i in q_2023[q_id]["Secondary_evidence_index"]])
    #
    #for q_id in tqdm(q_self_consistency_2024):
    #    res[f'{q_id}_self-consistency'] = q_self_consistency_2024[q_id]


    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    labels = json.load(open('../qrels/qrels2024_train.json'))
    labels_2 = json.load(open('../qrels/qrels2024_dev.json'))
    
    queries = json.load(open('../queries/queries2024_normal_self-consistency_evidence-index.json'))
    for q_id in tqdm(queries):
        res[f'{q_id}'] = queries[q_id]
        #print(q_id.split('_')[0])
        #print(labels_2[q_id.split('_')[0]]['Label'])
        res[f'{q_id}']['Label'] = labels[q_id.split('_')[0]]['Label'] if q_id.split('_')[0] in labels else labels_2[q_id.split('_')[0]]['Label']

        if "evidence" in q_id:
            prompt = f'Primary CTR Evidence: {res[q_id]["primary_evidence"]}'
            if "secondary_evidence" in res[q_id]:
                prompt += f'\nSecondary CTR Evidence: {res[q_id]["secondary_evidence"]}'
            prompt += f'\n\nFrom these evidences from CTRs, output the most important information to entail or contradict the following statement: "{res[q_id]["Statement"]}"\nOnly output information cointained in the evidence, and do not add any new information.'

            res[f'{q_id}']['Explanation'] = tokenize_generate_decode(model, tokenizer, prompt, 500)


    with safe_open_w(f'{args.output_name[:-5]}.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()