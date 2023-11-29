import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import time 
from data.data_processor import read_json_file, questions_from_wiki
from src.utils import import_model_and_tokenizer
from src.prompts import (BASELINE_PROMPT_WIKI, VERIFICATION_QUESTION_PROMPT_WIKI, 
                     EXECUTE_PLAN_PROMPT, FINAL_VERIFIED_PROMPT)

class WikiDataChainofVerificationTwoStep:
    def __init__(self, model_id, file_path, top_p, temperature):
        self.file_path = file_path
        self.data = read_json_file(file_path)
        self.model, self.tokenizer = import_model_and_tokenizer(model_id)
        self.questions = questions_from_wiki(self.data)
        self.top_p = top_p
        self.temperature = temperature
        
    def generate_tokens(self, prompt_tmpl, max_tokens):
        prompt = f"<|system|>\n</s>\n<|user|>\n{prompt_tmpl}</s>\n<|assistant|>\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
        tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
        tokens = tokens.split("<|assistant|>")[1]
        return tokens
    
    def run_verification_chain(self, original_question):
        baseline_prompt_tmpl = BASELINE_PROMPT_WIKI.format(original_question=original_question)
        baseline_tokens = self.generate_tokens(baseline_prompt_tmpl, 80)
        
        verification_question_prompt_tmpl = VERIFICATION_QUESTION_PROMPT_WIKI.format(original_question=original_question, baseline_response=baseline_tokens)
        verif_tokens = self.generate_tokens(verification_question_prompt_tmpl, 90)
        
        execute_questions_prompt_tmpl = EXECUTE_PLAN_PROMPT.format(verification_questions=verif_tokens)
        execute_verif_tokens = self.generate_tokens(execute_questions_prompt_tmpl, 100)
        
        final_verified_prompt_tmpl = FINAL_VERIFIED_PROMPT.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions=verif_tokens, verification_answers=execute_verif_tokens)
        final_verified_tokens = self.generate_tokens(final_verified_prompt_tmpl, 80)
        
        return baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens
    
    def print_results(self, original_question, baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens):
        print(f"Question: {original_question}\n")
        print(f"Baseline Answer: {baseline_tokens}\n")
        print(f"Verification Questions: {verif_tokens}\n")
        print(f"Execute Plan: {execute_verif_tokens}\n")
        print(f"Final Refined Answer: {final_verified_tokens}\n")
        print("---------------------------------------------------\n")
    
    def run_and_print_results(self):
        for q in self.questions[:30]:
            baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens = self.run_verification_chain(q)
            self.print_results(q, baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens)
            #time.sleep(2)

file_path = '/proj/layegh/users/x_amila/CoV/data/wikidata_questions.json'
llama_model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
zephyr_model_id = "HuggingFaceH4/zephyr-7b-beta"

wiki_chain_verifications = WikiDataChainofVerificationTwoStep(zephyr_model_id, file_path, 0.9, 0.7)
wiki_chain_verifications.run_and_print_results()
