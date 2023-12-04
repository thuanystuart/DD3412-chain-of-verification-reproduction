import json
import time 
from src.utils import import_model_and_tokenizer
from src.prompts import (BASELINE_PROMPT_WIKI, PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI, 
                     EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI, PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI, FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI
                     , FINAL_VERIFIED_JOINT_PROMPT_WIKI, BASELINE_PROMPT_MULTI_QA, PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA
                     , EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA, FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA,
                     BASELINE_PROMPT_WIKI_CATEGORY, PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY, EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
                     FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY)

class ChainofVerification():
    def __init__(self, model_id, top_p, temperature, dataset, setting, questions, access_token):
        self.model_id = model_id
        self.access_token = access_token
        self.model, self.tokenizer = import_model_and_tokenizer(model_id, access_token=self.access_token)
        self.questions = questions
        self.top_p = top_p
        self.temperature = temperature
        self.dataset = dataset
        self.setting = setting
        
    def generate_response(self, prompt, max_tokens):
        if self.model_id == "llama2" or self.model_id == "llama2_70b" or self.model_id == "llama-65b":
            #prompt = f"""<s>[INST] <<SYS>>{prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
            #tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
            tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][1:]
            tokens = tokens.split("[/INST]")[1]
        else:
            prompt = f"<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
            tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][1:]
            tokens = tokens.split("<|assistant|>")[1]
        return tokens
    
    def run_verification_chain(self, original_question):
        if self.dataset == "wikidata":
            baseline_prompt_tmpl = BASELINE_PROMPT_WIKI.format(original_question=original_question)
            baseline_prompt = f"""<s>[INST] <<SYS>>{baseline_prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            baseline_tokens = self.generate_response(baseline_prompt, 100)
            baseline_tokens = baseline_tokens.split("\n\n")[1]
            if self.setting == "two_step":
                verification_question_prompt_tmpl = PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI.format(original_question=original_question, baseline_response=baseline_tokens)
                verification_prompt = f"""<s>[INST] <<SYS>>{verification_question_prompt_tmpl}\n<</SYS>>\nVerification Questions: [/INST]"""
                verif_tokens = self.generate_response(verification_prompt, 180)
                verif_tokens = verif_tokens.split("\n\n")[1]
                
                execute_questions_prompt_tmpl = EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI.format(verification_questions=verif_tokens)
                execute_prompt = f"""<s>[INST] <<SYS>>{execute_questions_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
                execute_verif_tokens = self.generate_response(execute_prompt, 180)
                execute_verif_tokens = execute_verif_tokens.split("\n\n")[1]
                
                final_verified_prompt_tmpl = FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions=verif_tokens, verification_answers=execute_verif_tokens)
                final_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(final_prompt, 100)
                final_verified_tokens = final_verified_tokens.split("\n\n")[1]
                
                return baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens
            elif self.setting == "joint":
                plan_and_execution_prompt_tmpl = PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI.format(original_question=original_question, baseline_response=baseline_tokens)
                plan_and_execution_tokens = self.generate_response(plan_and_execution_prompt_tmpl, 150)
                if "Verification Questions and Answers:" in plan_and_execution_tokens:
                    plan_and_execution_tokens = plan_and_execution_tokens.split("Verification Questions and Answers:")[1]
                    if "Note:" in plan_and_execution_tokens:
                        plan_and_execution_tokens = plan_and_execution_tokens.split("Note:")[0]
                
                final_verified_prompt_tmpl = FINAL_VERIFIED_JOINT_PROMPT_WIKI.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions_and_answers=plan_and_execution_tokens)
                final_verified_tokens = self.generate_response(final_verified_prompt_tmpl, 100)
                
                return baseline_tokens, plan_and_execution_tokens, final_verified_tokens
            else:
                print("Invalid setting specified. Please specify either 'two_step' or 'joint'!----")
        elif self.dataset == "wikidata_category":
            baseline_prompt_tmpl = BASELINE_PROMPT_WIKI_CATEGORY.format(original_question=original_question)
            baseline_prompt = f"""<s>[INST] <<SYS>>{baseline_prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            baseline_tokens = self.generate_response(baseline_prompt, 100).split("\n\n")[1]
            if self.setting == "two_step":
                verification_question_prompt_tmpl = PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY.format(original_question=original_question, baseline_response=baseline_tokens)
                verification_prompt = f"""<s>[INST] <<SYS>>{verification_question_prompt_tmpl}\n<</SYS>>\nVerification Questions: [/INST]"""
                verif_tokens = self.generate_response(verification_prompt, 180).split("\n\n")[1]
                
                execute_questions_prompt_tmpl = EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY.format(verification_questions=verif_tokens)
                execute_prompt = f"""<s>[INST] <<SYS>>{execute_questions_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
                execute_verif_tokens = self.generate_response(execute_prompt, 180).split("\n\n")[1]
                
                final_verified_prompt_tmpl = FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions=verif_tokens, verification_answers=execute_verif_tokens)
                final_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(final_prompt, 100).split("\n\n")[1]
                
                return baseline_tokens, verif_tokens, execute_verif_tokens, final_verified_tokens
        elif self.dataset == "multi_qa":
            baseline_prompt_tmpl = BASELINE_PROMPT_MULTI_QA.format(original_question=original_question)
            baseline_tokens = self.generate_response(baseline_prompt_tmpl, 50)
            if self.setting == "two_step":
                plan_verification_prompt_tmpl = PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA.format(original_question=original_question, baseline_response=baseline_tokens)
                plan_verification_tokens = self.generate_response(plan_verification_prompt_tmpl, 100)
                if "some verification questions based on the baseline response:"  in plan_verification_tokens:
                    plan_verification_tokens = plan_verification_tokens.split("some verification questions based on the baseline response:")[1]
                
                execute_verification_prompt_tmpl = EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA.format(verification_questions=plan_verification_tokens)
                execute_verification_tokens = self.generate_response(execute_verification_prompt_tmpl, 140)
                if "Here are my answers:" in execute_verification_tokens:
                    execute_verification_tokens = execute_verification_tokens.split("Here are my answers:")[1]
                
                final_verified_prompt_tmpl = FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA.format(original_question=original_question, baseline_response=baseline_tokens, verification_questions_and_answers=execute_verification_tokens)
                final_verified_tokens = self.generate_response(final_verified_prompt_tmpl, 100)
                if "here's the final refined answer:" in final_verified_tokens:
                    final_verified_tokens = final_verified_tokens.split("here's the final refined answer:")[1]
                if "is:" in final_verified_tokens:
                    final_verified_tokens = final_verified_tokens.split("is:")[1]
                
                return baseline_tokens, plan_verification_tokens, execute_verification_tokens, final_verified_tokens
            else:
                print("Invalid setting specified. Please specify either 'two_step' or 'joint'!!!!!")
        else:
            print("Invalid dataset specified. Please specify either 'wikidata' or 'multi_qa'!")
            
    def run_and_store_results(self):
        all_results = []
        for q in self.questions:
            if self.setting == "two_step":
                baseline_tokens, plan_verification_tokens, execute_verification_tokens, final_verified_tokens = self.run_verification_chain(q)
                result_entry = {
                    "Question": q,
                    "Baseline Answer": baseline_tokens,
                    "Verification Questions": plan_verification_tokens,
                    "Execute Plan": execute_verification_tokens,
                    "Final Refined Answer": final_verified_tokens
                }
                print("----------------------\n")
                print(f"Question: {q}")
                print("----------------------\n")
                print(f"Baseline Answer: {baseline_tokens}")
                print("----------------------\n")
                print(f"Verification Questions: {plan_verification_tokens}")
                print("----------------------\n")
                print(f"Execute Plan: {execute_verification_tokens}")
                print("----------------------\n")
                print(f"Final Refined Answer: {final_verified_tokens}")
                print("=========================================\n")
                all_results.append(result_entry)
            elif self.setting == "joint":
                baseline_tokens, plan_and_execution_tokens, final_verified_tokens = self.run_verification_chain(q)
                result_entry = {
                    "Question": q,
                    "Baseline Answer": baseline_tokens,
                    "Plan and Execution": plan_and_execution_tokens,
                    "Final Refined Answer": final_verified_tokens
                }
                all_results.append(result_entry)
            else:
                print("Invalid setting specified. Please specify either 'two_step' or 'joint'!***")
        
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
        }

        if (self.model_id, self.dataset, self.setting) in valid_combinations:
            result_file_path = f'./results/{self.model_id}_{self.dataset}_{self.setting}_results.json'
            with open(result_file_path, 'w') as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
        else:
            print("Invalid model, dataset, or setting specified. Please check your inputs.")