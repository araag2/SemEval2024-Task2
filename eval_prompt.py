import json
import torch
import typing
import re

# Local Files
from utils import safe_open_w
from label_prompt_funcs import textlabel_2_binarylabel, label_2_SemEval2024, create_qid_prompt_label_dict, create_qdid_prompt

# Util libs
from datetime import datetime
from tqdm import tqdm
# Model libs
from sklearn.metrics import f1_score, precision_score, recall_score

def tokenize_and_generate(model : object, tokenizer : object, text : str) -> str:
        tokenized = tokenizer(text, return_tensors="pt")
        tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
        tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

        # We could use do_sample=False and disable top_k and top_p to get a deterministic output
        outputs =  model.generate(**tokenized, max_new_tokens=50, top_k = 5, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()

def query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            decoded_output = tokenize_and_generate(model, tokenizer, queries[q_id]["text"])
            decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)

            res_labels[q_id] = textlabel_2_binarylabel(decoded_output_sub.split(" "))
    return res_labels

def calculate_metrics(pred_labels : dict, gold_labels : dict) -> dict:
    res_labels = [[],[]]
    mistakes = []
    for q_id in pred_labels:
        res_labels[0].append(gold_labels[q_id]["gold_label"])
        res_labels[1].append(pred_labels[q_id])
        if res_labels[0][-1] != res_labels[1][-1]:
            mistakes.append({"q_id" : q_id, "gold_label" : res_labels[0][-1], "pred_label" : res_labels[1][-1]})

    precison_bin = precision_score(res_labels[0], res_labels[1])
    precision_micro = precision_score(res_labels[0], res_labels[1], average="micro")
    precision_macro = precision_score(res_labels[0], res_labels[1], average="macro")
    recall_bin = recall_score(res_labels[0], res_labels[1])
    recall_micro = recall_score(res_labels[0], res_labels[1], average="micro")
    recall_macro = recall_score(res_labels[0], res_labels[1], average="macro")
    f1_bin = f1_score(res_labels[0], res_labels[1])
    f1_micro = f1_score(res_labels[0], res_labels[1], average="micro")
    f1_macro = f1_score(res_labels[0], res_labels[1], average="macro")

    return {"precison_bin" :precison_bin, "precison_micro" : precision_micro, "precision_macro" : precision_macro, "recall_bin" : recall_bin,"recall_micro" : recall_micro, "recall_macro" : recall_macro, "f1_bin" : f1_bin, "f1_micro" : f1_micro, "f1_macro" : f1_macro}, mistakes

def output_mistakes(args : dict, mistakes : list, prompt : str, queries : dict, qrels : dict, used_set : str):
    # Output Mistakes
    mistakes = {
        "prompt" : prompt,
        "used_set" : used_set, 
        "mistake_stats" : {"Total" : len(mistakes), "Single" : 0, "Comparison" : 0, "Entailment" : 0, "Contradiction" : 0}, 
        "mistakes" : mistakes, 
        "og_queries" : {}
    }

    for dict_q_id in mistakes["mistakes"]:
        q_id = dict_q_id["q_id"]
        mistakes["og_queries"][q_id] = queries[q_id]
        mistakes["og_queries"][q_id]["gold_label"] = qrels[q_id]["Label"]
        mistakes["mistake_stats"][mistakes["og_queries"][q_id]["Type"]] += 1
        mistakes["mistake_stats"][qrels[q_id]["Label"]] += 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    with safe_open_w(f'{args.output_dir}mistakes/{timestamp}_{args.model.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(mistakes, ensure_ascii=False, indent=4))

def output_full_metrics(args : dict, prompt_id : str, full_prompt : str, used_set : str, metrics : dict):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results = {"timestamp" : timestamp}
    for arg in vars(args):
        results[arg] = getattr(args, arg)
    results["prompt"] = full_prompt
    results["set"] = used_set
    results["metrics"] = metrics
    results["formated_metrics"] =f'| {args.model.split("/")[-1]}-{prompt_id}   | {metrics["f1_macro"]} | {metrics["precision_macro"]} | {metrics["recall_macro"]} | - |'

    with safe_open_w(f'{args.output_dir}combination_output/{timestamp}_{args.model.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def full_evaluate_prompt(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics, mistakes = calculate_metrics(pred_labels, queries_dict)
    output_mistakes(args, mistakes, prompt, queries, qrels, used_set)
    
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)
    return metrics

def output_prompt_labels(model : object, tokenizer : object, queries : dict, prompt : str, args : object, used_set : str):
    # Replace prompt with query info
    queries_dict = create_qdid_prompt(queries, prompt)

    # 0-shot inference from queries
    pred_labels = query_inference(model, tokenizer, queries_dict)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(label_2_SemEval2024(pred_labels), ensure_ascii=False, indent=4))

def output_prompt_res(model : object, tokenizer : object, queries : dict, qrels : str, prompt : str, args : object, used_set : str):
    # Replace prompt with query info
    queries = create_qid_prompt_label_dict(queries, qrels, prompt)

    with torch.inference_mode():
        for q_id in tqdm(queries):
            queries[q_id]["expanded_text"] = tokenize_and_generate(model, tokenizer, queries[q_id]["text"])
            queries[q_id]["text"] += queries[q_id]["expanded_text"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(queries, ensure_ascii=False, indent=4))