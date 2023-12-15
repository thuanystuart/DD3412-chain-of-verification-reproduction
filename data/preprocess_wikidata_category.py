import argparse
import json
from typing import List
import jsonlines


def read_data(path: str) -> json:
    with jsonlines.open(path) as json_file:
        json_data = [obj for obj in json_file]
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


def format_question(question: str) -> str:
    question = (
        question.replace("what are some ", "")
        .replace("what are ", "")
        .replace("list some ", "")
    )
    return "Name some " + question


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i",
        "--data-path",
        type=str,
        help="Path to the input dataset.",
        default="./dataset/source/wikidata_category.jsonl",
    )
    argParser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to the output dataset.",
        default="./dataset/wikidata_category_dataset.json",
    )
    args = argParser.parse_args()

    data = read_data(args.data_path)

    # select answer template to be non-logical
    data = list(filter(lambda sample: sample["metadata"]["template"] == "_", data))
    data = list(filter(lambda sample: " or " not in sample["query"], data))

    # filter for short answers
    data = list(
        filter(
            lambda sample: len(sample["docs"]) >= 6 and len(sample["docs"]) <= 10, data
        )
    )

    # format question and answer
    data = list(
        map(
            lambda sample: {
                "question": format_question(sample["query"]),
                "answer": sample["docs"],
            },
            data,
        )
    )

    num_items = {}
    for sample in data:
        l = len(sample["answer"])
        if l in num_items:
            num_items[l] += 1
        else:
            num_items[l] = 1
    num_items = list(num_items.items())
    num_items = sorted(num_items, key=lambda item: item[0])

    print(f"Number of selected samples: {len(data)}")
    save_data(data, args.output_path)
