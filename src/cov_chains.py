import json
import sys
from typing import Dict
from src.utils import (
    TaskConfig,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)

class ChainOfVerification:
    def __init__(self, model_id, task, setting):
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
        

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_baseline_response(self, question: str) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def run_two_step_chain(self, question: str, baseline_response: str):
       raise NotImplementedError("Subclasses must implement this method.")

    def run_joint_chain(self, question: str, baseline_response: str):
        raise NotImplementedError("Subclasses must implement this method.")

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

        result_file_path = (
            f"./result/{self.model_id}_{self.task}_{self.setting}_results.json"
        )
        with open(result_file_path, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)
