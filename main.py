import argparse
from dotenv import dotenv_values

from src.utils import get_absolute_path
from data.data_processor import (
    read_json,
    get_questions_from_list,
    get_questions_from_dict,
)


CONFIG = dotenv_values(get_absolute_path(".configurations"))
hf_access_token = CONFIG.get("HF_API_KEY")
openain_access_token = CONFIG.get("OPENAI_API_KEY")

file_path_mapping = {
    "wikidata": get_absolute_path("dataset/wikidata_questions.json"),
    "multispanqa": get_absolute_path("dataset/multispanqa_dataset.json"),
    "wikidata_category": get_absolute_path("dataset/wikidata_category_dataset.json"),
}

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m",
        "--model",
        type=str,
        help="LLM to use for predictions.",
        default="llama2",
        choices=["llama2", "llama2_70b", "llama-65b", "gpt3"],
    )
    argParser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task.",
        default="wikidata",
        choices=["wikidata", "wikidata_category", "multispanqa"],
    )
    argParser.add_argument(
        "-s",
        "--setting",
        type=str,
        help="Setting.",
        default="joint",
        choices=["joint", "two_step", "factored"],
    )
    argParser.add_argument(
        "-temp", "--temperature", type=float, help="Temperature.", default=0.07
    )
    argParser.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)
    args = argParser.parse_args()

    data = read_json(file_path_mapping[args.task])
    if args.task == "wikidata":
        questions = get_questions_from_dict(data)
    else:
        questions = get_questions_from_list(data)

    if args.model == "gpt3":
        from src.cove_chains_openai import ChainOfVerificationOpenAI
        chain_openai = ChainOfVerificationOpenAI(
            model_id=args.model,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            openai_access_token=openain_access_token,
        )
        chain_openai.run_chain()
    else:
        from src.cove_chains_hf import ChainOfVerificationHuggingFace
        chain_hf = ChainOfVerificationHuggingFace(
            model_id=args.model,
            top_p=args.top_p,
            temperature=args.temperature,
            task=args.task,
            setting=args.setting,
            questions=questions,
            hf_access_token=hf_access_token,
        )
        chain_hf.run_chain()
