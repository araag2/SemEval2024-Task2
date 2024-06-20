import os
import json

def safe_open_w(path: str) -> object:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def create_path(path : str) -> None:
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

## These stat function can easily be refactored, but for now I'll leave them as they are since they are not crucial to any functionality

def dataset_stats(dataset_path : str) -> None:
    dataset = json.load(open(dataset_path, encoding="utf8"))

    samples = len(dataset)
    labels = {"Entailment": 0, "Contradiction": 0}
    types = {"Single" : 0, "Comparison" : 0}
    sections = {"Adverse Events" : 0, "Eligibility" : 0, "Intervention" : 0, "Results" : 0}
    interventions_done = {"Paraphrase" : 0, "Contradiction" : 0, "Text_appended" : 0, "Numerical_contradiction" : 0, "Numerical_paraphrase" :0}
    type_of_intervention = {"Preserving" : 0, "Altering" : 0}

    for query in dataset:
        for stat in ["Type", "Section_id", "Intervention", "Label"]:
            if stat in dataset[query]:
                types[dataset[query][stat]] += 1

        if "Causal_type" in dataset[query]:
            type_of_intervention[dataset[query]["Causal_type"][0]] += 1

    output_prints = [f"# of samples: {samples}"]
    for stats in [labels, types, sections, interventions_done, type_of_intervention]:
        output_prints.append("")
        for key in stats:
            output_prints[-1]+=(f"{key}: {stats[key]} ({stats[key]/samples*100:.2f}%) | ")
        output_prints[-1] = output_prints[-1][:-3]

    for line in output_prints:
        print(line)

def count_corpus_stats(corpus_path : str) -> None:
    corpus = json.load(open(corpus_path, encoding="utf8"))
    corpus_words = sum([len(" ".join(corpus[doc][sec]).split(" ")) for doc in corpus for sec in corpus[doc]])
    print(f"Corpus size: {len(corpus)}")
    print(f"Corpus words: {corpus_words} | {corpus_words//len(corpus)} words per document")

def check_mistakes(results_path : str, qrels_path : str) -> None:
    results = json.load(open(results_path, encoding="utf8"))
    qrels = json.load(open(qrels_path, encoding="utf8"))

    total_errors, base_errors, intervention_errors = 0, 0, {}

    for query in results:
        wrong = results[query]["Prediction"] != qrels[query]["Label"]

        if "Intervention" not in qrels[query]:
            base_errors += 1 if wrong else 0

        else:
            if qrels[query]["Intervention"] not in intervention_errors:
                intervention_errors[qrels[query]["Intervention"]] = {"total" : 0, "wrong" : 0}
            intervention_errors[qrels[query]["Intervention"]]["total"] += 1
            intervention_errors[qrels[query]["Intervention"]]["wrong"] += 1 if wrong else 0

            if qrels[query]["Causal_type"][0] not in intervention_errors:
                intervention_errors[qrels[query]["Causal_type"][0]] = {"total" : 0, "wrong" : 0}
            intervention_errors[qrels[query]["Causal_type"][0]]["total"] += 1
            intervention_errors[qrels[query]["Causal_type"][0]]["wrong"] += 1 if wrong else 0

        total_errors += 1 if wrong else 0

    print(f"Total errors: {total_errors} in {len(results)} samples ({total_errors/len(results)*100:.2f}%)")
    print(f"No intervention errors: {base_errors} ({base_errors/total_errors*100:.2f}%)")
    print(f"Intervention errors: {total_errors-base_errors} ({(total_errors-base_errors)/total_errors*100:.2f}%)")
    for error in intervention_errors:
        print(f"{error}: {intervention_errors[error]['wrong']} / {intervention_errors[error]['total']} ({intervention_errors[error]['wrong']/intervention_errors[error]['total']*100:.1f}\%)")