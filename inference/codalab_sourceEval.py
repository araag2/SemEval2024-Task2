#!/usr/bin/env python3

import json
import os
import os.path
import sys
import warnings
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter('ignore')


def extract_control_set(predictions, gold):
    control_predicitons = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            control_predicitons[key] = predictions[key]
    return control_predicitons


def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]
    return para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions


def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering


def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness


def consistency(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] == gold[key]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency


def extract_contrast_set(predictions, gold):
    contrast_predicitons = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predicitons[key] = predictions[key]
    return contrast_predicitons


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = precision_score(gold_labels, pred_labels)
    Precision = recall_score(gold_labels, pred_labels)
    return F1, Recall, Precision


def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')


def calc_scores(predictions : dict, predictions_name : str, output_dir : str):
    gold = json.load(open('corpus/SemEval2024_gold-test.json'))

    # Control Test Set F1, Recall, Precision PUBLIC
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(extract_control_set(predictions, gold), gold)


    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, gold)


    # Intervention-wise Consistency & Faithfullness HIDDEN
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    para_Consistency = consistency(para_preserving, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, gold)
    definitions_Consistency = consistency(definitions_preserving, gold)


    # Intervention-wise F1, Recall, Precision HIDDEN
    Contrast_F1, Contrast_Rec, Contrast_Prec = F1_Recall_Precision(contrast_predictions, gold)
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)
    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(numerical_para_predictions, gold)
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(numerical_cont_predictions, gold)
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(definitions_predictions, gold)

    with safe_open_w(f'{output_dir}SCORES-{predictions_name[:-5]}.txt') as f:
        print(f'Scores for file: {predictions_name}\n', file=f)

        print(f'Overall Control Score\n', file=f)
        print(f'Control_F1: {Control_F1:.3f}', file=f)
        print(f'Control_Precision: {Control_Prec:.3f}', file=f)
        print(f'Control_Recall: {Control_Rec:.3f}', file=f)

        print(f'\nOverall Contrast Score', file=f)
        print(f'Contrast_F1: {Contrast_F1:.3f}', file=f)
        print(f'Contrast_Precision: {Contrast_Prec:.3f}', file=f)
        print(f'Contrast_Recall: {Contrast_Rec:.3f}', file=f)
        print(f'Faithfulness: {Faithfulness:.3f}', file=f)
        print(f'Consistency: {Consistency:.3f}', file=f)

        print(f'\nParaphrase Scores', file=f)
        print(f'\nText Paraphrase', file=f)
        print(f'Para_F1: {para_F1:.3f}', file=f)
        print(f'Para_Precision: {para_Prec:.3f}', file=f)
        print(f'Para_Recall: {para_Rec:.3f}', file=f)
        print(f'Para_Consistency: {para_Consistency:.3f}\n', file=f)

        print(f'Numerical Paraphrase', file=f)
        print(f'Numerical_Para_F1: {numerical_para_F1:.3f}', file=f)
        print(f'Numerical_Para_Precision: {numerical_para_Prec:.3f}', file=f)
        print(f'Numerical_Para_Recall: {numerical_para_Rec:.3f}', file=f)
        print(f'Numerical_Para_Consistency: {numerical_para_Consistency:.3f}\n', file=f)


        print(f'Contradiction Scores', file=f)
        print(f'Text Contradiction', file=f)
        print(f'Cont_F1: {cont_F1:.3f}', file=f)
        print(f'Cont_Precision: {cont_Prec:.3f}', file=f)
        print(f'Cont_Recall: {cont_Rec:.3f}', file=f)
        print(f'Cont_Faithfulness: {cont_Faithfulness:.3f}', file=f)
        print(f'Cont_Consistency: {cont_Consistency:.3f}\n', file=f)

        print(f'Numerical Contradiction', file=f)
        print(f'Numerical_Cont_F1: {numerical_cont_F1:.3f}', file=f)
        print(f'Numerical_Cont_Precision: {numerical_cont_Prec:.3f}', file=f)
        print(f'Numerical_Cont_Recall: {numerical_cont_Rec:.3f}', file=f)
        print(f'Numerical_Cont_Faithfulness: {numerical_cont_Faithfulness:.3f}', file=f)
        print(f'Numerical_Cont_Consistency: {numerical_cont_Consistency:.3f}\n', file=f)

        print(f'Text_Append Scores', file=f)
        print(f'Text_Append_F1: {definitions_F1:.3f}', file=f)
        print(f'Text_Append_Precision: {definitions_Prec:.3f}', file=f)
        print(f'Text_Append_Recall: {definitions_Rec:.3f}', file=f)
        print(f'Text_Append_Consistency: {definitions_Consistency:.3f}', file=f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_res', type=str, help='path to input res file with labels for each query', default='')
    parser.add_argument('--output_dir', type=str, help='path to output file with scores', default="outputs/")
    args = parser.parse_args()

    predictions = json.load(open(args.input_res))
    calc_scores(predictions, args.input_res.split('/')[-1], args.output_dir)


if '__main__' == __name__:
    main()