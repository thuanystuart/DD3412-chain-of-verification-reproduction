import json
import sys
from typing import Dict
from data.data_processor import get_items_from_answer
from src.utils import (
    TaskConfig,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)


class ChainOfVerification:
    def __init__(self, model_id, task, setting, questions):
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

        self.questions = questions

    def call_llm(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def process_prompt(self, prompt, command) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_response(self, prompt: str, max_tokens: int, command) -> str:
        processed_prompt = self.process_prompt(prompt, command)
        return self.call_llm(processed_prompt, max_tokens)

    def get_baseline_response(self, question: str) -> str:
        baseline_prompt = self.task_config.baseline_prompt.format(
            original_question=question
        )
        return self.generate_response(
            prompt=baseline_prompt,
            max_tokens=self.task_config.max_tokens,
            command=self.task_config.baseline_command,
        )

    def run_two_step_chain(self, question: str, baseline_response: str):
        # Create Plan
        plan_prompt = self.task_config.two_step.plan_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
        )

        plan_response = self.generate_response(
            prompt=plan_prompt,
            max_tokens=self.task_config.two_step.max_tokens_plan,
            command=self.task_config.two_step.plan_command,
        )

        ## Execute Plan
        execute_prompt = self.task_config.two_step.execute_prompt.format(
            verification_questions=plan_response
        )

        execute_response = self.generate_response(
            prompt=execute_prompt,
            max_tokens=self.task_config.two_step.max_tokens_execute,
            command=self.task_config.two_step.execute_command,
        )

        ## Verify
        verify_prompt = self.task_config.two_step.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions=plan_response,
            verification_answers=execute_response,
        )

        verify_response = self.generate_response(
            prompt=verify_prompt,
            max_tokens=self.task_config.two_step.max_tokens_verify,
            command=self.task_config.two_step.verify_command,
        )

        return (
            plan_response,
            execute_response,
            verify_response,
        )

    def run_joint_chain(self, question: str, baseline_response: str):
        ## Create and Execute Plan
        plan_and_execution_prompt = (
            self.task_config.joint.plan_and_execute_prompt.format(
                original_question=question,
                baseline_response=baseline_response,
            )
        )

        plan_and_execution_response = self.generate_response(
            prompt=plan_and_execution_prompt,
            max_tokens=self.task_config.joint.max_tokens_plan_and_execute,
            command=self.task_config.joint.plan_and_execute_command,
        )

        ## Verify
        verify_prompt = self.task_config.joint.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions_and_answers=plan_and_execution_response,
        )

        verify_response = self.generate_response(
            prompt=verify_prompt,
            max_tokens=self.task_config.joint.max_tokens_verify,
            command=self.task_config.joint.verify_command,
        )

        return plan_and_execution_response, verify_response

    def run_factored_chain(self, question: str, baseline_response: str):
        ## Create Plan
        plan_prompt = self.task_config.factored.plan_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
        )
        plan_response = self.generate_response(
            prompt=plan_prompt,
            max_tokens=self.task_config.factored.max_tokens_plan,
            command=self.task_config.factored.plan_command,
        )

        ## Execute Plan
        planned_questions = get_items_from_answer(plan_response)
        execute_responses = []
        for planned_question in planned_questions:
            execute_prompt = self.task_config.factored.execute_prompt.format(
                verification_question=planned_question
            )
            execute_response = self.generate_response(
                prompt=execute_prompt,
                max_tokens=self.task_config.factored.max_tokens_execute,
                command=self.task_config.factored.execute_command,
            )
            execute_responses.append(execute_response)
        execute_response = "\n".join(
            [
                f"{i+1}. {execute_response}"
                for i, execute_response in enumerate(execute_responses)
            ]
        )

        ## Verify
        verify_prompt = self.task_config.factored.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions=plan_response,
            verification_answers=execute_response,
        )
        verify_response = self.generate_response(
            prompt=verify_prompt,
            max_tokens=self.task_config.factored.max_tokens_verify,
            command=self.task_config.factored.verify_command,
        )

        return (
            plan_response,
            execute_response,
            verify_response,
        )

    def print_result(self, result: Dict[str, str]):
        for key, value in result.items():
            print(f"{key}: {value}")
            print("----------------------\n")
        print("=========================================\n")

    def run_chain(self):
        all_results = []
        for question in self.questions:
            baseline_response = self.get_baseline_response(question)
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
            elif self.setting == "factored":
                (
                    plan_verification_tokens,
                    execute_verification_tokens,
                    final_verified_tokens,
                ) = self.run_factored_chain(question, baseline_response)
                result = {
                    "Question": question,
                    "Baseline Answer": baseline_response,
                    "Verification Questions": plan_verification_tokens,
                    "Execute Plan": execute_verification_tokens,
                    "Final Refined Answer": final_verified_tokens,
                }
                self.print_result(result)
                all_results.append(result)

        result_file_path = (
            f"./result/{self.model_id}_{self.task}_{self.setting}_results.json"
        )
        with open(result_file_path, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)
