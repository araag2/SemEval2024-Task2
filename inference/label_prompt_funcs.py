import json
import typing

# Local Files
from .utils import safe_open_w

ENTAILMENT_LABELS = {"entailment", "yes", "y", "yes.", "(yes)"}
CONTRADICTION_LABELS = {"contradiction", "not", "no", "n", "no.", "(no)"}

def textlabel_2_binarylabel(text_label: list[str]) -> int:
    for label in text_label:
        if label.lower() in ENTAILMENT_LABELS:
            return 1
        elif label.lower() in CONTRADICTION_LABELS:
            return 0
    return 1 # In case of no label, default to Entailment

def label_2_SemEval2024(labels : dict) -> dict:
    return {q_id : {"Prediction" : "Entailment" if labels[q_id] == 1 else "Contradiction"} for q_id in labels}

# Base queries and prompts

TASK_TYPES = {"base" : [], "self_consistency" : ["entail", "contradict"], "explain_answer" : ["explain"], "section_info" : ["Primary_Section_Info", "Secondary_Section_Info"]}

def extract_info_from_query(query : dict, task_type : str = "base") -> dict:
    relevant_info = {}
    relevant_info["hypothesis"] = query["Statement"]
    relevant_info["primary_evidence"] = query["Primary_id_txt"]
    relevant_info["secondary_evidence"] = query["Secondary_id_txt"] if "Secondary_id_txt" in query else ""
    if task_type in TASK_TYPES:
        for field in TASK_TYPES[task_type]:
            #TODO: Check which field is the correct one
            relevant_info[field] = query[field] if field in query else ""
    return relevant_info

def generate_query_from_prompt(text_to_replace: dict, prompt: str, task_type : str = "base") -> str:
    prompt = prompt.replace("$primary_evidence", text_to_replace["primary_evidence"])
    prompt = prompt.replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    prompt = prompt.replace("$hypothesis", text_to_replace["hypothesis"])
    if task_type in TASK_TYPES:
        for field in TASK_TYPES[task_type]:
            prompt = prompt.replace(f"${field}", text_to_replace[field])
    return prompt

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompt : str, task_type : str = "base") -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = { 
            "text" : generate_query_from_prompt(extract_info_from_query(queries[q_id], task_type), prompt, task_type), 
            "gold_label" : textlabel_2_binarylabel([qrels[q_id]["Label"].strip()])
        }
    return queries_dict

def create_qdid_prompt(queries : dict, prompt : str, task_type : str = "base") -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {"text" : generate_query_from_prompt(extract_info_from_query(queries[q_id], task_type), prompt, task_type)}
    return queries_dict

def generate_pos_prompts(mistral_prompts : dict):
    prompt_combinations = { "base_mistral_prompts" : {field : mistral_prompts[field] for field in mistral_prompts}, "combination_prompts" : {}}

    for task_id, task in mistral_prompts["task_descriptions"].items():
        for ctr_id, ctr in mistral_prompts["ctr_descriptions"].items():
            for statement_id, statement in mistral_prompts["statement_descriptions"].items():
                for option_id, option in mistral_prompts["option_descriptions"].items():
                    combination = mistral_prompts["task_template_prompt_comparison"].replace("$task_description", task).replace("$ctr_description", ctr).replace("$statement_description", statement).replace("$option_description", option)

                    prompt_combinations["combination_prompts"][f'<s>[INST]{task_id}_{ctr_id}_{statement_id}_{option_id}[/INST]'] = combination

    with safe_open_w(f'prompts/MistralPromptsCombination_V2.json') as output_file:
        output_file.write(json.dumps(prompt_combinations, ensure_ascii=False, indent=4))

    return prompt_combinations