import json
from typing import Dict, List
import re

def read_jsonlines(path: str) -> Dict[str, str]:
    records = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def read_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_questions_from_dict(data: Dict[str, str]) -> List[str]:
    return list(data.keys())

def get_questions_from_list(data: List[Dict[str, str]]) -> List[str]:
    questions = [entry['question'] for entry in data]
    return questions

def get_answers_from_dict(data: Dict[str, str]) -> List[str]:
    return list(data.items())

def get_answers_from_list(data: List[Dict[str, str]]) -> List[str]:
    answers = [entry['answer'] for entry in data]
    return answers

def get_items_from_answer(result: str) -> List[str]:
    answers = result.split("\n")
    answers = [re.sub("([\d.]*\d+)\.\ ", "", answer) for answer in answers]
    return answers

def get_cleaned_final_answer(results: List[str], answer_slice: str) -> List[List[str]]:
    return [get_items_from_answer(result[answer_slice]) for result in results]
