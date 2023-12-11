from src.cov_chains_hf import ChainOfVerificationHuggingFace
from src.cov_chains_openai import ChainOfVerificationOpenAI
from data.data_processor import (
    read_json,
    get_questions_from_list,
    get_questions_from_dict,
)
from src.utils import get_absolute_path

from dotenv import dotenv_values
CONFIG = dotenv_values(get_absolute_path(".configurations"))

gpt_35_id = "gpt3"
llama2_id = "llama2"
llama_2_70b_id = "llama2_70b"
llama_id = "llama-65b"

hf_access_token = CONFIG["HF_API_KEY"]
openain_access_token = CONFIG["OPENAI_API_KEY"]

file_path_wikidata = get_absolute_path("dataset/wikidata_questions.json")
file_path_multispan = get_absolute_path("dataset/multispanqa_dataset.json")
file_path_wikidata_categories = get_absolute_path("dataset/wikidata_category_dataset.json")

wikidata_dataset = "wikidata"
multi_qa_dataset = "multi_qa"
wikidata_category_dataset = "wikidata_category"

task = wikidata_dataset

valid_combinations = {
    ("llama2", "wikidata", "two_step"),
    ("llama2", "wikidata", "joint"),
    ("llama2", "multi_qa", "two_step"),
    ("llama2_70b", "wikidata", "two_step"),
    ("llama2_70b", "wikidata", "joint"),
    ("llama2_70b", "multi_qa", "two_step"),
    ("llama-65b", "wikidata", "two_step"),
    ("llama-65b", "wikidata", "joint"),
    ("llama-65b", "multi_qa", "two_step"),
    ("llama2", "wikidata_category", "two_step"),
}


data = read_json(file_path_wikidata)
if task == wikidata_dataset:
    questions = get_questions_from_dict(data)
else:
    questions = get_questions_from_list(data)

chain_hf = ChainOfVerificationHuggingFace(
    model_id=llama2_id,
    top_p=0.9,
    temperature=0.07,
    task=task,
    setting="two_step",
    questions=questions,
    hf_access_token=hf_access_token,
)
chain_hf.run_chain()


chain_openai = ChainOfVerificationOpenAI(
    model_id=gpt_35_id,
    temperature=0.07,
    task=task,
    setting="two_step",
    questions=questions,
    openai_access_token=openain_access_token,
)
chain_openai.run_chain()
