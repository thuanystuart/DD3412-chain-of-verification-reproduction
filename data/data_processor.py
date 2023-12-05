import json
from typing import Dict, List

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