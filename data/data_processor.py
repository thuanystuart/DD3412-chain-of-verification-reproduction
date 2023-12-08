import json
from typing import Dict, List
import re

def read_jsonlines(path: str):
    records = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def read_json(path: str):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_questions_from_dict(data: Dict[str, str]):
    return list(data.keys())

def get_questions_from_list(data: List[Dict[str, str]]):
    questions = [entry['question'] for entry in data]
    return questions

def get_answers_from_dict(data: Dict[str, str]):
    return list(data.items())

def get_answers_from_list(data: List[Dict[str, str]]):
    answers = [entry['answer'] for entry in data]
    return answers

def get_cleaned_final_answer(results: List[str], answer_slice: str) -> List[List[str]]:
    def clean_result(result: str):
        answers = result.split("\n")
        answers = [re.sub("([\d.]*\d+)\.\ ", "", answer) for answer in answers]
        return answers

    return [clean_result(result[answer_slice]) for result in results]
