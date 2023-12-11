import argparse
import json
from typing import List


def read_data(path: str) -> json:
    with open(path, encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        return json_data


def save_data(data: json, path: str) -> json:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_answer(label: List[str], context: List[str]) -> List[str]:
    answer = []
    for mask, word in zip(label, context):
        if mask in ["I", "B"]:
            answer.append(word)
    return answer


def format_question(question: List[str]) -> str:
    return " ".join([question[0].title()] + question[1:]) + "?"


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i",
        "--data-path",
        type=str,
        help="Path to the input dataset.",
        default="./dataset/source/multispanqa.json",
    )
    argParser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to the output dataset.",
        default="./dataset/multispanqa_dataset.json",
    )
    args = argParser.parse_args()

    data = read_data(args.data_path)
    data = data["data"]

    # select answer type: human, location, and numeric
    data = list(filter(lambda sample: sample["type"] in ["HUM", "LOC", "NUM"], data))

    # get question-answer and select answer spans from label and context
    data = list(
        map(
            lambda sample: {
                "question": sample["question"],
                "answer": parse_answer(sample["label"], sample["context"]),
            },
            data,
        )
    )

    # filter for short answers <= 3 tokens
    data = list(filter(lambda sample: len(sample["answer"]) <= 3, data))

    # format question and answer
    data = list(
        map(
            lambda sample: {
                "question": format_question(sample["question"]),
                "answer": " ".join(sample["answer"]),
            },
            data,
        )
    )

    print(f"Number of selected samples: {len(data)}")
    save_data(data, args.output_path)
