import argparse
from typing import List
import numpy as np
from data import read_json, get_cleaned_final_answer, get_answers_from_dict, get_answers_from_list

def compute_metrics_for_list_answer(answers: List[List[str]], true_answers: List[List[str]]):
    positive_answers = []
    negative_answers = []
    for answer, true_answer in zip(answers, true_answers):
        positive = 0
        negative = 0
        for item in answer:
            if item in true_answer:
                positive += 1
            else:
                negative += 1
        positive_answers.append(positive)
        negative_answers.append(negative)

    tp = np.sum(positive_answers)
    fp = np.sum(negative_answers)

    return {
        "positive_avg": np.mean(positive_answers),
        "negative_avg": np.mean(negative_answers),
        "precision": tp / (tp+fp),
    }

def evaluate(result_path: str, dataset_path: str, dataset_type: str):
    dataset = read_json(dataset_path)
    results = read_json(result_path)

    if dataset_type == "wikidata":
        true_answers = get_answers_from_dict(dataset)
    elif dataset_type == "wikidata_category":
        true_answers = get_answers_from_list(dataset)

    answers = get_cleaned_final_answer(results, "Final Refined Answer")
    baseline_answers = get_cleaned_final_answer(results, "Baseline Answer")

    baseline_metrics = compute_metrics_for_list_answer(baseline_answers, true_answers)
    metrics = compute_metrics_for_list_answer(answers, true_answers)

    print(f"Baseline metrics: {baseline_metrics}")
    print(f"CoVe metrics: {metrics}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-r", "--result-path", type=str, help="Path to the result file.")
    argParser.add_argument("-d", "--dataset-path", type=str, help="Path to the original dataet.")
    argParser.add_argument("-t", "--dataset-type", type=str, help="Type of the dataet.")

    args = argParser.parse_args()

    evaluate(args.result_path, args.dataset_path, args.dataset_type)
