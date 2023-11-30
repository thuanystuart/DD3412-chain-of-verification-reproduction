from src import cov_chains, prompts, utils
from src.cov_chains import WikiDataChainofVerificationTwoStep
from data import *

file_path = 'PATH_TO_DATA_FILE'
llama_model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
zephyr_model_id = "HuggingFaceH4/zephyr-7b-beta"
#mistral_instruction_model_id = "mrm8488/mistral-7b-ft-h4-no_robots_instructions"
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
llama2_model_id = "meta-llama/Llama-2-13b-chat-hf"

wiki_chain_verifications = WikiDataChainofVerificationTwoStep(llama2_model_id, file_path, 0.9, 0.7)
wiki_chain_verifications.run_and_print_results()