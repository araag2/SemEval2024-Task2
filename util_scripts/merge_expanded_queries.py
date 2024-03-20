import os
import json
import argparse
import torch

from transformers import AutoModelForCausalLM
from peft import PeftModel

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-og_queries', type=str, default='../qrels/qrels2024_train.json')
    parser.add_argument('-ex_queries', type=str, default='../outputs/2024-03-20_12-07_train-set.json')
    args = parser.parse_args()

    og_q = json.load(open(args.og_queries))
    ex_q = json.load(open(args.ex_queries))

    res = {}

    for q_id in ex_q:
        res[f'{q_id}_appended-text'] = og_q[q_id]
        res[f'{q_id}_appended-text']["Statement"] += " " + ex_q[q_id]["expanded_text"][:-4] 

    with safe_open_w(f'{args.og_queries[:-5]}_appended-text.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()