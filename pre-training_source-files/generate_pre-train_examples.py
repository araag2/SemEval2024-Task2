import os
import json
import argparse
import torch

section_2_id = {"Eligibility"      : ["$eligibility_criteria", "(1) ELIGIBILITY CRITERIA"],
                "Intervention"   : ["$interventions", "(2) INTERVENTION"],
                "Results"        : ["$results", "(3) RESULTS"],
                "Adverse Events" : ["$adverse_events", "(4) ADVERSE EVENTS"]
}

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def subs_ctr_info(ctr : str, prompt : str, section_to_complete : str) -> str:
    res = ""
    for section in section_2_id:
        if section != section_to_complete:
            prompt = prompt.replace(section_2_id[section][0], "\n".join(ctr[section]))
        else:
            prompt = prompt.replace(section_2_id[section][0], "")
            res = "\n".join(ctr[section])

    return prompt.replace("$missing_section_name", section_2_id[section_to_complete][1]), res

def filter_eligibility_criteria(ctr : str, prompt : str, inclusion_or_exclusion : str, line_to_ignore : int) -> str:
    prompt = prompt.replace("$eligibility_criteria_segment", "\n".join([ctr[line] if line != line_to_ignore else "" for line in range(len(ctr))]))
    return prompt.replace("$exclusion_or_inclusion", inclusion_or_exclusion), ctr[line_to_ignore]
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, default='../corpus/SemEval_CT-corpus.json')
    parser.add_argument('--output_p', type=str, default='task_prompts/')
    parser.add_argument('--task_type', type=str, default='complete_ctr-info', choices=["complete_ctr-info", "complete_elegibility-criteria"])
    args = parser.parse_args()

    corpus = json.load(open(args.corpus, encoding='utf8'))
    prompt = json.load(open("task_prompts/pre-train_macro-prompts.json", encoding="utf8"))[args.task_type]
    res = {}

    if args.task_type == "complete_ctr-info":
        for ctr in corpus:
            for section in corpus[ctr]:
                #res[f'{ctr}_complete-{section}'] = {"text" : "", "ground_truth" : ""}
                #res[f'{ctr}_complete-{section}']["text"], res[f'{ctr}_complete-{section}']["ground_truth"], = subs_ctr_info(corpus[ctr], prompt, section)
                res[f'{ctr}_complete-{section}'] = {"id" : f'{ctr}_complete-{section}', "text" : ""}
                res[f'{ctr}_complete-{section}']["text"] = f'{subs_ctr_info(corpus[ctr], prompt, section)[0]} {subs_ctr_info(corpus[ctr], prompt, section)[1]}'

    elif args.task_type == "complete_elegibility-criteria":
        for ctr in corpus:
            curr = "Inclusion Criteria"
            for line in range(1, len(corpus[ctr]["Eligibility"])):
                if corpus[ctr]["Eligibility"][line].lower() == "exclusion criteria":
                    curr = "Exclusion Criteria"
                    break

                #res[f'{ctr}_complete-{curr}_{line}'] = {"text" : "", "ground_truth" : ""}
                #res[f'{ctr}_complete-{curr}_{line}']["text"], res[f'{ctr}_complete-{curr}_{line}']["ground_truth"], = filter_eligibility_criteria(corpus[ctr]["Eligibility"], prompt, curr, line)

                res[f'{ctr}_complete-{curr}_{line}'] = {"id" : f'{ctr}_complete-{curr}_{line}', "text" : ""}
                res[f'{ctr}_complete-{curr}_{line}']["text"] = f'{filter_eligibility_criteria(corpus[ctr]["Eligibility"], prompt, curr, line)[0]} {filter_eligibility_criteria(corpus[ctr]["Eligibility"], prompt, curr, line)[1]}'

    elif args.task_type == "explantions":
        

    print(f'Res Size: {len(res)}, Average Size per ctr: {len(res) / len(corpus)}')

    with safe_open_w(f'{args.output_p}pre-train_{args.task_type}.json') as out_f:
        json.dump(res, out_f, indent=4)

if __name__ == '__main__':
    main()