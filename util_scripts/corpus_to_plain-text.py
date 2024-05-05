import json
import os
import argparse

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus', type=str, default='../corpus/SemEval_CT-corpus.json')
    parser.add_argument('--task_type', type=str, default='Eligibility')
    args = parser.parse_args()

    corpus = json.load(open(args.corpus))

    res = {"id" : [], "text" : []}

    if args.task_type == 'CT_to-text':

        sections = { "Eligibility" : "(1) ELIGIBILITY CRITERIA delineating conditions for patient inclusion",
                    "Intervention" : "(2) INTERVENTIONS particulars specifying type, dosage, frequency, and duration of treatments",
                    "Results" : "(3) RESULTS summary encompassing participant statistics, outcome measures, units, and conclusions",
                    "Adverse Events" : "(4) ADVERSE EVENTS cataloging signs and symptoms observed"}

        for ctr_id in corpus:
            for section in sections:
                res["id"].append(f'{ctr_id}_{section}')
                res["text"].append(f'{sections[section]}:\n{chr(10).join(corpus[ctr_id][section])}')

    elif args.task_type == 'Eligibility':
        for ctr_id in corpus:
            eligibility = "\n".join(corpus[ctr_id]["Eligibility"])
            split_eligibility = eligibility.split("Exclusion Criteria:")

            res["id"].append(ctr_id+"_Inclusion")
            res["text"].append(split_eligibility[0])

            if len(split_eligibility) > 1:
                res["id"].append(ctr_id+"_Exclusion")
                res["text"].append("Exclusion Criteria:"+split_eligibility[1])


    with safe_open_w("../corpus/SemEval_criteria-for-mlm.json") as f:
        json.dump(res, f, indent=4)

if __name__ == '__main__':
    main()