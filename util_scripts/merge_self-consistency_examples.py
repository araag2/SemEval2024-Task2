import os
import json
import argparse

from tqdm import tqdm

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='explain_e-c')
    parser.add_argument('--og_queries', type=str, default='../queries/queries2024_test.json')
    parser.add_argument('--entail_queries', type=str, default='../outputs/entailment2024-03-25_15-40_dev-set.json')
    parser.add_argument('--contradict_queries', type=str, default='../outputs/contradiction2024-03-25_15-38_dev-set.json')
    args = parser.parse_args()

    og_q = json.load(open(args.og_queries))
    entail_q = json.load(open(args.entail_queries))
    contradict_q = json.load(open(args.contradict_queries))

    res = {}

    for q_id in tqdm(og_q):
        res[f'{q_id}'] = og_q[q_id]
        res[f'{q_id}']["explain"] = entail_q[q_id]["expanded_text"][:-4] 
        #res[f'{q_id}']["entail"] = entail_q[q_id]["expanded_text"] 
        #res[f'{q_id}']["contradict"] = contradict_q[q_id]["expanded_text"]

    with safe_open_w(f'{args.og_queries[:-5]}_{args.exp_name}.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()