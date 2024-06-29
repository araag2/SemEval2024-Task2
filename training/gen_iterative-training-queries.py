import os
import json
import argparse

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_q', type=str, default='queries/queries2024_train.json')
    parser.add_argument('--prompt', type=str, default='prompts/DPOPrompts.json')
    parser.add_argument('--explanations_queries', type=str, default='outputs/explain_query-102024-06-28_03-30_train-set.json')

    parser.add_argument('--output_name', type=str, default='iteration-1_iterative-train-explanations')
    args = parser.parse_args()

    base_queries = json.load(open(args.base_q, encoding='utf8'))

    prompt = json.load(open(args.prompt, encoding="utf8"))["answer_with_explanations"]

    explanations_queries = json.load(open(args.explanations_queries, encoding='utf8'))
    res = {}

    for query_id in explanations_queries:
        l_prompt = prompt

        l_prompt = l_prompt.replace("$primary_evidence", base_queries[query_id]["Primary_id_txt"])
        l_prompt = l_prompt.replace("$hypothesis", base_queries[query_id]["Statement"])


        if "Secondary_id" not in base_queries[query_id]:
            l_prompt = l_prompt.replace("\n\nSecondary Trial:\n\n$secondary_evidence", "")
        else:
            l_prompt = l_prompt.replace("$secondary_evidence", base_queries[query_id]["Secondary_id_txt"])

        for i in range(len(explanations_queries[query_id]["expanded_text"])):
            res[f"{query_id}_{i}"] = {
                "id": f"{query_id}_{i}",
                "text": f'{l_prompt.replace("$explanation", explanations_queries[query_id]["expanded_text"][i])}{"NO" if explanations_queries[query_id]["gold_label"] == 0 else "YES"}',
            }

    with safe_open_w(f'training/iterative_training_queries/{args.output_name}.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()