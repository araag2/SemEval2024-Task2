import os
import json
import argparse

from tqdm import tqdm

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, help='name of the experiment', default='run-7_ensemble_self-consistency_top-5')
    args = parser.parse_args()

    files_list = ["../outputs/run-7_self-consistency_top-5_02024-04-18_12-54_test_train-self-consistency_top-5-set.json", "../outputs/run-7_self-consistency_top-5_12024-04-18_12-55_test_train-self-consistency_top-5-set.json", "../outputs/run-7_self-consistency_top-5_22024-04-18_12-56_test_train-self-consistency_top-5-set.json", "../outputs/run-7_self-consistency_top-5_32024-04-18_12-57_test_train-self-consistency_top-5-set.json", "../outputs/run-7_self-consistency_top-5_42024-04-18_12-58_test_train-self-consistency_top-5-set.json"]
    num_files = len(files_list)

    queries_dicts = [json.load(open(file)) for file in files_list]

    res = {}

    for q_id in tqdm(queries_dicts[0]):
        entails = [queries_dicts[i][q_id]["Prediction"] == "Entailment" for i in range(num_files)]
        res[q_id] = { "Prediction" : "Entailment" if sum(entails) > num_files / 2 else "Contradiction"}

    with safe_open_w(f'../outputs/{args.exp_name}.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()