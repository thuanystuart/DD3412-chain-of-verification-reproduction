import json

def read_json_file(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

def questions_from_wiki(data):
    questions = []
    for record in data:
        for i in range (0, len(record)):
            question = list(record.keys())[i]
            questions.append(question)
    return questions

def read_questions_from_multi_qa_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)
    questions = [entry['question'] for entry in data]
    return questions