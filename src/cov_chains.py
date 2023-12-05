import json
import sys
from src.utils import import_model_and_tokenizer, MODEL_MAPPING, Model
from src.prompts import (
    BASELINE_PROMPT_WIKI,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI,
    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI,
    FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI,
    FINAL_VERIFIED_JOINT_PROMPT_WIKI,
    BASELINE_PROMPT_MULTI_QA,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
    FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA,
    BASELINE_PROMPT_WIKI_CATEGORY,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
    FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY,
    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI_CATEGORY,
    FINAL_VERIFIED_JOINT_PROMPT_WIKI_CATEGORY,
)


class ChainofVerification:
    def __init__(
        self, model_id, top_p, temperature, task, setting, questions, access_token
    ):
        self.model_id = model_id
        self.model: Model = MODEL_MAPPING[model_id]
        if self.model is None:
            print("Invalid Model ID. Please write either 'mistral', 'llama2' or 'zephyr'.")
            sys.exit()

        self.access_token = access_token
        self.questions = questions
        self.top_p = top_p
        self.temperature = temperature
        self.task = task
        self.setting = setting

        self.model, self.tokenizer = import_model_and_tokenizer(
            self.model, access_token=self.access_token
        )

    def generate_response(self, prompt, max_tokens):
        if self.model.is_llama:
            # prompt = f"""<s>[INST] <<SYS>>{prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).input_ids.cuda()
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            # tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
            tokens = self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0][0:]
            tokens = tokens.split("[/INST]")[1]  # comment for multiqa
        # else:
        #     prompt = f"<|system|>\n</s>\n<|user|>\n{prompt_tmpl}</s>\n<|assistant|>\n"
        #     input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        #     outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
        #     tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
        #     tokens = tokens.split("<|assistant|>")[1]
        return tokens

    def run_verification_chain(self, original_question):
        if self.task == "wikidata":
            baseline_prompt_tmpl = BASELINE_PROMPT_WIKI.format(
                original_question=original_question
            )
            baseline_prompt = f"""<s>[INST] <<SYS>>{baseline_prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            baseline_tokens = self.generate_response(baseline_prompt, 150)
            baseline_tokens = baseline_tokens.split("\n\n")[1]
            if self.setting == "two_step":
                verification_question_prompt_tmpl = (
                    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                    )
                )
                verification_prompt = f"""<s>[INST] <<SYS>>{verification_question_prompt_tmpl}\n<</SYS>>\nVerification Questions: [/INST]"""
                verif_tokens = self.generate_response(verification_prompt, 300).split(
                    "\n\n"
                )[1]
                # verif_tokens = verif_tokens.split("\n\n")[1]

                execute_questions_prompt_tmpl = (
                    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI.format(
                        verification_questions=verif_tokens
                    )
                )
                execute_prompt = f"""<s>[INST] <<SYS>>{execute_questions_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
                execute_verif_tokens = self.generate_response(execute_prompt, 300)
                execute_verif_tokens = execute_verif_tokens.split("\n\n")[1]

                final_verified_prompt_tmpl = FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI.format(
                    original_question=original_question,
                    baseline_response=baseline_tokens,
                    verification_questions=verif_tokens,
                    verification_answers=execute_verif_tokens,
                )
                final_verified_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(
                    final_verified_prompt, 300
                )
                final_verified_tokens = final_verified_tokens.split("\n\n")[1]

                return (
                    baseline_tokens,
                    verif_tokens,
                    execute_verif_tokens,
                    final_verified_tokens,
                )
            elif self.setting == "joint":
                plan_and_execution_prompt_tmpl = (
                    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                    )
                )
                plan_and_execution_prompt = f"""<s>[INST] <<SYS>>{plan_and_execution_prompt_tmpl}\n<</SYS>>\nVerification Questions and Answers: [/INST]"""
                plan_and_execution_tokens = self.generate_response(
                    plan_and_execution_prompt, 500
                ).split("\n\n")[1]
                # if "Verification Questions and Answers:" in plan_and_execution_tokens:
                #     plan_and_execution_tokens = plan_and_execution_tokens.split("Verification Questions and Answers:")[1]
                #     if "Note:" in plan_and_execution_tokens:
                #         plan_and_execution_tokens = plan_and_execution_tokens.split("Note:")[0]

                final_verified_prompt_tmpl = FINAL_VERIFIED_JOINT_PROMPT_WIKI.format(
                    original_question=original_question,
                    baseline_response=baseline_tokens,
                    verification_questions_and_answers=plan_and_execution_tokens,
                )
                final_verified_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(
                    final_verified_prompt, 150
                ).split("\n\n")[1]

                return baseline_tokens, plan_and_execution_tokens, final_verified_tokens
            else:
                print(
                    "Invalid setting specified. Please specify either 'two_step' or 'joint'!----"
                )
        elif self.task == "wikidata_category":
            baseline_prompt_tmpl = BASELINE_PROMPT_WIKI_CATEGORY.format(
                original_question=original_question
            )
            baseline_prompt = f"""<s>[INST] <<SYS>>{baseline_prompt_tmpl}\n<</SYS>>\nAnswer: [/INST]"""
            baseline_tokens = self.generate_response(baseline_prompt, 100).split(
                "\n\n"
            )[1]
            if self.setting == "two_step":
                verification_question_prompt_tmpl = (
                    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                    )
                )
                verification_prompt = f"""<s>[INST] <<SYS>>{verification_question_prompt_tmpl}\n<</SYS>>\nVerification Questions: [/INST]"""
                verif_tokens = self.generate_response(verification_prompt, 180).split(
                    "\n\n"
                )[1]

                execute_questions_prompt_tmpl = (
                    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY.format(
                        verification_questions=verif_tokens
                    )
                )
                execute_prompt = f"""<s>[INST] <<SYS>>{execute_questions_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
                execute_verif_tokens = self.generate_response(
                    execute_prompt, 180
                ).split("\n\n")[1]

                final_verified_prompt_tmpl = (
                    FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                        verification_questions=verif_tokens,
                        verification_answers=execute_verif_tokens,
                    )
                )
                final_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(final_prompt, 100).split(
                    "\n\n"
                )[1]

                return (
                    baseline_tokens,
                    verif_tokens,
                    execute_verif_tokens,
                    final_verified_tokens,
                )
            elif self.setting == "joint":
                plan_and_execution_prompt_tmpl = (
                    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI_CATEGORY.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                    )
                )
                plan_and_execution_prompt = f"""<s>[INST] <<SYS>>{plan_and_execution_prompt_tmpl}\n<</SYS>>\nVerification Questions and Answers: [/INST]"""
                plan_and_execution_tokens = self.generate_response(
                    plan_and_execution_prompt, 400
                ).split("\n\n")[1]

                final_verified_prompt_tmpl = (
                    FINAL_VERIFIED_JOINT_PROMPT_WIKI_CATEGORY.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                        verification_questions_and_answers=plan_and_execution_tokens,
                    )
                )
                final_verified_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(
                    final_verified_prompt, 100
                ).split("\n\n")[1]

                return baseline_tokens, plan_and_execution_tokens, final_verified_tokens
        elif self.task == "multi_qa":
            baseline_prompt_tmpl = BASELINE_PROMPT_MULTI_QA.format(
                original_question=original_question
            )
            baseline_prompt = f"""<s>[INST] <<SYS>>{baseline_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
            baseline_tokens = self.generate_response(baseline_prompt, 200).split(
                "\n\n"
            )[1]
            if self.setting == "two_step":
                plan_verification_prompt_tmpl = (
                    PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                    )
                )
                plan_verification_prompt = f"""<s>[INST] <<SYS>>{plan_verification_prompt_tmpl}\n<</SYS>>\nVerification Questions: [/INST]"""
                plan_verification_tokens = self.generate_response(
                    plan_verification_prompt, 400
                ).split("\n\n")[1]
                # if "some verification questions based on the baseline response:"  in plan_verification_tokens:
                #     plan_verification_tokens = plan_verification_tokens.split("some verification questions based on the baseline response:")[1]

                execute_verification_prompt_tmpl = (
                    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA.format(
                        verification_questions=plan_verification_tokens
                    )
                )
                execute_verification_prompt = f"""<s>[INST] <<SYS>>{execute_verification_prompt_tmpl}\n<</SYS>>\nAnswers: [/INST]"""
                execute_verification_tokens = self.generate_response(
                    execute_verification_prompt, 400
                ).split("\n\n")[1]
                # if "Here are my answers:" in execute_verification_tokens:
                #     execute_verification_tokens = execute_verification_tokens.split("Here are my answers:")[1]

                final_verified_prompt_tmpl = (
                    FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA.format(
                        original_question=original_question,
                        baseline_response=baseline_tokens,
                        verification_questions=plan_verification_tokens,
                        verification_answers=execute_verification_tokens,
                    )
                )
                final_verified_prompt = f"""<s>[INST] <<SYS>>{final_verified_prompt_tmpl}\n<</SYS>>\nFinal Refined Answer: [/INST]"""
                final_verified_tokens = self.generate_response(
                    final_verified_prompt, 300
                ).split("\n\n")[1]
                # if "here's the final refined answer:" in final_verified_tokens:
                #     final_verified_tokens = final_verified_tokens.split("here's the final refined answer:")[1]
                # if "is:" in final_verified_tokens:
                #     final_verified_tokens = final_verified_tokens.split("is:")[1]

                return (
                    baseline_tokens,
                    plan_verification_tokens,
                    execute_verification_tokens,
                    final_verified_tokens,
                )
            else:
                print(
                    "Invalid setting specified. Please specify either 'two_step' or 'joint'!!!!!"
                )
        else:
            print(
                "Invalid dataset specified. Please specify either 'wikidata' or 'multi_qa'!"
            )

    def run_and_store_results(self):
        all_results = []
        for q in self.questions:
            if self.setting == "two_step":
                (
                    baseline_tokens,
                    plan_verification_tokens,
                    execute_verification_tokens,
                    final_verified_tokens,
                ) = self.run_verification_chain(q)
                result_entry = {
                    "Question": q,
                    "Baseline Answer": baseline_tokens,
                    "Verification Questions": plan_verification_tokens,
                    "Execute Plan": execute_verification_tokens,
                    "Final Refined Answer": final_verified_tokens,
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
                (
                    baseline_tokens,
                    plan_and_execution_tokens,
                    final_verified_tokens,
                ) = self.run_verification_chain(q)
                result_entry = {
                    "Question": q,
                    "Baseline Answer": baseline_tokens,
                    "Plan and Execution": plan_and_execution_tokens,
                    "Final Refined Answer": final_verified_tokens,
                }
                all_results.append(result_entry)
            else:
                print(
                    "Invalid setting specified. Please specify either 'two_step' or 'joint'!***"
                )

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
            ("llama2_70b", "wikidata_category", "two_step"),
            ("llama2_70b", "wikidata_category", "joint"),
        }

        if (self.model_id, self.task, self.setting) in valid_combinations:
            result_file_path = (
                f"./results/{self.model_id}_{self.task}_{self.setting}_results.json"
            )
            with open(result_file_path, "w") as json_file:
                json.dump(all_results, json_file, indent=2, ensure_ascii=False)
        else:
            print(
                "Invalid model, dataset, or setting specified. Please check your inputs."
            )
