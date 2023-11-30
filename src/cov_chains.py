import torch
import data
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import time 
from src.utils import import_model_and_tokenizer
from src.prompts import (BASELINE_PROMPT_WIKI, VERIFICATION_QUESTION_PROMPT_WIKI, 
                     EXECUTE_PLAN_PROMPT, FINAL_VERIFIED_PROMPT)
#from data.data_processor import read_json_file, questions_from_wiki

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

class WikiDataChainofVerificationTwoStep:
    def __init__(self, model_id, file_path, top_p, temperature):
        self.file_path = file_path
        self.model_id = model_id
        self.data = read_json_file(file_path)
        self.model, self.tokenizer = import_model_and_tokenizer(model_id)
        self.questions = questions_from_wiki(self.data)
        self.top_p = top_p
        self.temperature = temperature
        
    def generate_tokens(self, prompt_tmpl, max_tokens):
        if "meta-llama" in self.model_id: # meta-llama models prompts need to be wrapped in [INST] and [/INST]
            prompt = f"""<s>[INST] <<SYS>>\n<</SYS>>\n{prompt_tmpl} [/INST]"""
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
            tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
            tokens = tokens.split("[/INST]")[1]
        else:
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
        verif_tokens = self.generate_tokens(verification_question_prompt_tmpl, 110)
        
        execute_questions_prompt_tmpl = EXECUTE_PLAN_PROMPT.format(verification_questions=verif_tokens)
        execute_verif_tokens = self.generate_tokens(execute_questions_prompt_tmpl, 100)
        
        final_verified_prompt_tmpl = FINAL_VERIFIED_PROMPT.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions=verif_tokens, verification_answers=execute_verif_tokens)
        final_verified_tokens = self.generate_tokens(final_verified_prompt_tmpl, 80)
        
        return baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens
    
    def print_results(self, original_question, baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens):
        
        print(f"Question: {original_question}\n")
        print("---------------------------------------------------\n")
        print(f"Baseline Answer: {baseline_tokens}\n")
        print("---------------------------------------------------\n")
        print(f"Verification Questions: {verif_tokens}\n")
        print("---------------------------------------------------\n")
        print(f"Execute Plan: {execute_verif_tokens}\n")
        print("---------------------------------------------------\n")
        print(f"Final Refined Answer: {final_verified_tokens}\n")
        print("=======================================================\n")
    
    def run_and_print_results(self):
        all_results = []
        for q in self.questions:
            baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens = self.run_verification_chain(q)
            result_entry = {
                "Question": q,
                "Baseline Answer": baseline_tokens,
                "Verification Questions": verif_tokens,
                "Execute Plan": execute_verif_tokens,
                "Final Refined Answer": final_verified_tokens
            }
            all_results.append(result_entry)
            self.print_results(q, baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens)
        
        if "meta-llama" in self.model_id:
            with open('./results/llama2_wiki_results.json', 'w') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
        if "HuggingFaceH4/zephyr-7b-beta" in self.model_id:
            with open('./results/zephyr_wiki_results.json', 'w') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
        else:
            with open('./lm_wiki_results.json', 'w') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
