import json
import torch
import typing
import re

# Local Files
from .utils import safe_open_w
from .label_prompt_funcs import textlabel_2_binarylabel, label_2_SemEval2024, create_qid_prompt_label_dict, create_qdid_prompt

# Util libs
from datetime import datetime
from tqdm import tqdm
# Model libs
from sklearn.metrics import f1_score, precision_score, recall_score

#Constraint Decoding
from genre.trie import MarisaTrie

class MyMarisaTrie(MarisaTrie):
    def __init__(self, data): super().__init__(data)
    def get(self, data, tokenizer, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])

def tokenize_generate_decode(model : object, tokenizer : object, text : str, max_new_tokens : int = 50, top_k : int = 50, top_p : float = 0.95, do_sample : bool = True, temperature : float = 1.0) -> str:   
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    # We could use do_sample=False and disable top_k and top_p to get a deterministic output
    outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens, top_k = top_k, top_p = top_p, do_sample=do_sample, temperature = temperature, pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()

def tokenize_generate_five_decode(model : object, tokenizer : object, text : str, max_new_tokens : int = 50, top_k : int = 5, top_p : float = 0, do_sample : bool = True, temperature : float = 1.0) -> str:   
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    # We could use do_sample=False and disable top_k and top_p to get a deterministic output
    outputs = model.generate(**tokenized, max_new_tokens=max_new_tokens, top_k = top_k, top_p = top_p, do_sample=do_sample, temperature = temperature, pad_token_id=tokenizer.eos_token_id, num_return_sequences= 5)

    return [re.sub("(</s>)*", "", tokenizer.decode(out[tokenized["input_ids"].shape[1]:]).strip()) for out in outputs]

def tokenize_generate_decode_constraint(model : object, tokenizer : object, text : str, trie : object) -> str:
    tokenized = tokenizer(text, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

    #outputs =  model.generate(**tokenized, pad_token_id=tokenizer.eos_token_id, max_new_tokens = 2, do_sample = True, top_k = 10)
    #print(f'Tensors -> {outputs[0][tokenized["input_ids"].shape[1]:]} that decode to {tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()}')

    outputs =  model.generate(**tokenized, pad_token_id=tokenizer.eos_token_id, max_new_tokens = 3, prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), tokenizer, tokenized["input_ids"].shape[1]))
    return tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()

def query_inference(model : object, tokenizer : object, queries : dict, constraint : bool = False) -> dict:
    res_labels = {}
    
    #trie = MarisaTrie([ [0]+tokenizer.encode(‘Yes’) , [0]+tokenizer.encode(‘No’)])

    # Tokens for Yes and No
    trie = MyMarisaTrie([[7929, 28723], [7929,  13], [7929, 28725], [627, 2255]]) if constraint else None

    with torch.inference_mode():
        for q_id in tqdm(queries):
            decoded_output = ""
            if not constraint:
                decoded_output = tokenize_generate_decode(model, tokenizer, queries[q_id]["text"], 5, 15, 0.70, True)
            else:
                decoded_output = tokenize_generate_decode_constraint(model, tokenizer, queries[q_id]["text"], trie)
                print(f'Query Output: -> {decoded_output}')

            decoded_output_sub = re.sub("[,!\.()-]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)

            #print(decoded_output_sub)

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

def full_evaluate_prompt(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, args : object, used_set : str, task_type : str = "base") -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt, task_type)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict, constraint=True)

    # Compute metrics
    metrics, mistakes = calculate_metrics(pred_labels, queries_dict)
    output_mistakes(args, mistakes, prompt, queries, qrels, used_set)
    
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)
    return metrics

def output_prompt_labels(model : object, tokenizer : object, queries : dict, prompt : str, args : object, used_set : str, constraint : bool = False, task_type : str = "base"):
    # Replace prompt with query info
    queries_dict = create_qdid_prompt(queries, prompt, task_type)

    # 0-shot inference from queries
    pred_labels = query_inference(model, tokenizer, queries_dict, constraint)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    exp_name = args.exp_name if "exp_name" in args else ""

    # Output results
    with safe_open_w(f'{args.output_dir}{exp_name if exp_name != "" else args.checkpoint}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(label_2_SemEval2024(pred_labels), ensure_ascii=False, indent=4))

def output_prompt_res(model : object, tokenizer : object, queries : dict, qrels : str, prompt : str, args : object, used_set : str, task_type : str = "base"):
    # Replace prompt with query info
    queries = create_qid_prompt_label_dict(queries, qrels, prompt, task_type)

    with torch.inference_mode():
        for q_id in tqdm(queries):
            #TO:DO Change this afterwards
            queries[q_id]["expanded_text"] = tokenize_generate_decode(model, tokenizer, queries[q_id]["text"], 500, 15, 0.70, True)
            #queries[q_id]["expanded_text"] = tokenize_generate_five_decode(model, tokenizer, queries[q_id]["text"], 500, 50, 0.95, True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{args.exp_name if "exp_name" in args else ""}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(queries, ensure_ascii=False, indent=4))