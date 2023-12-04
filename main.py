from src.cov_chains import ChainofVerification
from data.data_processor import read_json_file, questions_from_wiki, read_questions_from_multi_qa_dataset

mistral_id = "mistral"
llama2_id = "llama2"
zephyr_id = "zephyr"
llama_2_70b_id = "llama2_70b"
llama_id = "llama-65b"
access_token = "YOUR_ACCESS_TOKEN"
file_path = "PATH_TO_THE_FILE"

file_path = 'PATH_TO_WIKIDATA_DATASET'
file_path_multi = 'PATH_TO_MULTI_QA_DATASET'
file_path_wikidata_categories = 'PATH_TO_WIKIDATA_CATEGORIES_DATASET'

wikidata_dataset = "wikidata"
multi_qa_dataset = "multi_qa"
wikidata_category_dataset = "wikidata_category"
access_token = "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"

dataset = wikidata_category_dataset

valid_combinations = {
            ("llama2", "wikidata", "two_step"),
            ("llama2", "wikidata", "joint"),
            ("llama2", "multi_qa", "two_step"),
            ("zephyr", "wikidata", "two_step"),
            ("zephyr", "wikidata", "joint"),
            ("zephyr", "multi_qa", "two_step"),
            ("mistral", "wikidata", "two_step"),
            ("mistral", "wikidata", "joint"),
            ("mistral", "multi_qa", "two_step"),
            ("llama2_70b", "wikidata", "two_step"),
            ("llama2_70b", "wikidata", "joint"),
            ("llama2_70b", "multi_qa", "two_step"),
            ("llama-65b", "wikidata", "two_step"),
            ("llama-65b", "wikidata", "joint"),
            ("llama-65b", "multi_qa", "two_step"),
            ("llama2", "wikidata_category", "two_step"),
        }


if dataset == wikidata_dataset:
    data= read_json_file(file_path)
    questions = questions_from_wiki(data)
elif dataset == multi_qa_dataset:
    questions = read_questions_from_multi_qa_dataset(file_path_multi)
elif dataset == wikidata_category_dataset:
    questions = read_questions_from_multi_qa_dataset(file_path_wikidata_categories)

chain = ChainofVerification(model_id=llama2_id, top_p=0.9, temperature=0.07, dataset=dataset, setting="two_step", questions=questions, access_token=access_token)
chain.run_and_store_results()