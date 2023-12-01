from src import cov_chains, prompts, utils
from src.cov_chains import WikiDataChainofVerificationTwoStep, WikiDataChainofVerificationJoint, ChainofVerification
from data.data_processor import read_json_file, questions_from_wiki, read_questions_from_multi_qa_dataset

mistral_id = "mistral"
llama2_id = "llama2"
zephyr_id = "zephyr"
access_token = "YOUR_ACCESS_TOKEN"
file_path = "PATH_TO_THE_FILE"

wikidata_dataset = "wikidata"
multi_qa_dataset = "multi_qa"

dataset = multi_qa_dataset

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
        }


if dataset == wikidata_dataset:
    data= read_json_file(file_path)
    questions = questions_from_wiki(data)
else:
    questions = read_questions_from_multi_qa_dataset(file_path)
chain = ChainofVerification(model_id=llama2_id, top_p=0.9, temperature=0.07, dataset=dataset, setting="two_step", questions=questions, access_token=access_token)
chain.run_and_store_results()