from typing import Any, Dict, List, Optional
import json

from cove_chain_factored import *
from cove_chain_two_step import *

class WikiDataQAPair:
    def __init__(self, question: str, answer: List[str]):
        self.question = question
        self.answer = answer

    def __str__(self) -> str:
        return f"Question: {self.question}\nAnswer: {', '.join(self.answer)}\n"
    
OPENAI_API_KEY = "" ## REPLACE THIS LINE WITH PROPER API KEY
WIKIDATA_DATASET_FILE_PATH = 'wikidata_dataset.json'

# Read the JSON file
with open(WIKIDATA_DATASET_FILE_PATH, 'r', encoding='utf-8') as file:
    data = json.load(file)

qa_pairs: List[WikiDataQAPair] = [WikiDataQAPair(item['question'], item['answer']) for item in data]


chain_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo-0613",
    max_tokens=500
)

wiki_data_category_list_cove_chain_instance_factored = WikiDataCategoryListCOVEChainFactored(chain_llm)
wiki_data_category_list_cove_chain_factored = wiki_data_category_list_cove_chain_instance_factored()

wiki_data_category_list_cove_chain_instance_two_step = WikiDataCategoryListCOVEChainTwoStep(chain_llm)
wiki_data_category_list_cove_chain_two_step = wiki_data_category_list_cove_chain_instance_two_step()


question = qa_pairs[0].question


two_step_out = wiki_data_category_list_cove_chain_two_step({"original_question": question})

factored_out = wiki_data_category_list_cove_chain_factored({"original_question": question})