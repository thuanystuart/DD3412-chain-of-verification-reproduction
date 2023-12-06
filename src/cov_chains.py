import json
import sys
from typing import Dict
from src.utils import (
    TaskConfig,
    import_model_and_tokenizer,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)


class ChainOfVerification:
    def __init__(
        self, model_id, top_p, temperature, task, setting, questions, access_token
    ):
        self.model_id = model_id
        self.model_config: ModelConfig = MODEL_MAPPING.get(model_id, None)
        if self.model_config is None:
            print(f"Invalid model. Valid models are: {', '.join(MODEL_MAPPING.keys())}")
            sys.exit()

        self.task = task
        self.task_config: TaskConfig = TASK_MAPPING.get(task, None)
        if self.task_config is None:
            print(f"Invalid task. Valid taks are: {', '.join(TASK_MAPPING.keys())}")
            sys.exit()

        self.setting = setting
        if self.setting not in SETTINGS:
            print(f"Invalid setting. Valid settings are: {', '.join(SETTINGS)}")
            sys.exit()
        if self.task_config.__dict__[self.setting] is None:
            print(
                f"Invalid combination. Settings {self.setting} was not implemented for task {self.task}"
            )
            sys.exit()

        self.access_token = access_token
        self.questions = questions
        self.top_p = top_p
        self.temperature = temperature

        self.model, self.tokenizer = import_model_and_tokenizer(
            self.model_config, access_token=self.access_token
        )

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        if self.model_config.is_llama:
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
            tokens = self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0][0:]
            if self.task_config.id != "multi_qa":
                tokens = tokens.split("[/INST]")[1]
        # else:
        #     prompt = f"<|system|>\n</s>\n<|user|>\n{prompt_tmpl}</s>\n<|assistant|>\n"
        #     input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        #     outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
        #     tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
        #     tokens = tokens.split("<|assistant|>")[1]
        return tokens.split("\n\n")[1]

    def get_baseline_tokens(self, question: str) -> str:
        baseline_prompt = self.task_config.baseline_prompt.format(
            original_question=question
        )
        baseline_prompt = self.model_config.prompt_format.format(
            prompt=baseline_prompt, command=self.task_config.baseline_command
        )
        return self.generate_response(baseline_prompt, self.task_config.max_tokens)

    def run_two_step_chain(self, question: str, baseline_response: str):
        plan_prompt = self.task_config.two_step.plan_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
        )
        plan_prompt = self.model_config.prompt_format.format(
            prompt=plan_prompt, command=self.task_config.two_step.plan_command
        )
        plan_response = self.generate_response(
            plan_prompt, self.task_config.two_step.max_tokens_plan
        )

        execute_prompt = self.task_config.two_step.execute_prompt.format(
            verification_questions=plan_response
        )
        execute_prompt = self.model_config.prompt_format.format(
            prompt=execute_prompt, command=self.task_config.two_step.execute_command
        )
        execute_response = self.generate_response(
            execute_prompt, self.task_config.two_step.max_tokens_execute
        )

        verify_prompt = self.task_config.two_step.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions=plan_response,
            verification_answers=execute_response,
        )
        verify_prompt = self.model_config.prompt_format.format(
            prompt=verify_prompt, command=self.task_config.two_step.verify_command
        )
        verify_response = self.generate_response(
            verify_prompt, self.task_config.two_step.max_tokens_verify
        )

        return (
            plan_response,
            execute_response,
            verify_response,
        )

    def run_joint_chain(self, question: str, baseline_response: str):
        plan_and_execution_prompt = (
            self.task_config.joint.plan_and_execute_joint_prompt.format(
                original_question=question,
                baseline_response=baseline_response,
            )
        )
        plan_and_execution_prompt = self.model_config.prompt_format.format(
            prompt=plan_and_execution_prompt,
            command=self.task_config.joint.plan_and_execute_command,
        )
        plan_and_execution_response = self.generate_response(
            plan_and_execution_prompt,
            self.task_config.joint.max_tokens_plan_and_execute,
        )

        verify_prompt = self.task_config.joint.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions_and_answers=plan_and_execution_response,
        )
        verify_prompt = self.model_config.prompt_format.format(
            prompt=verify_prompt, command=self.task_config.joint.verify_command
        )
        verify_response = self.generate_response(
            verify_prompt, self.task_config.joint.max_tokens_verify
        )

        return plan_and_execution_response, verify_response

    def print_result(self, result: Dict[str, str]):
        for key, value in result.items():
            print(f"{key}: {value}")
            print("----------------------\n")
        print("=========================================\n")

    def run_chain(self):
        all_results = []
        for question in self.questions:
            baseline_response = self.get_baseline_tokens(question)
            if self.setting == "two_step":
                (
                    plan_verification_tokens,
                    execute_verification_tokens,
                    final_verified_tokens,
                ) = self.run_two_step_chain(question, baseline_response)
                result = {
                    "Question": question,
                    "Baseline Answer": baseline_response,
                    "Verification Questions": plan_verification_tokens,
                    "Execute Plan": execute_verification_tokens,
                    "Final Refined Answer": final_verified_tokens,
                }
                self.print_result(result)
                all_results.append(result)
            elif self.setting == "joint":
                (
                    plan_and_execution_tokens,
                    final_verified_tokens,
                ) = self.run_joint_chain(question, baseline_response)
                result = {
                    "Question": question,
                    "Baseline Answer": baseline_response,
                    "Plan and Execution": plan_and_execution_tokens,
                    "Final Refined Answer": final_verified_tokens,
                }
                self.print_result(result)
                all_results.append(result)

        result_file_path = (
            f"./results/{self.model_id}_{self.task}_{self.setting}_results.json"
        )
        with open(result_file_path, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)
