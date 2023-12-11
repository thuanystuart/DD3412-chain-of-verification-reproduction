from src.cov_chains import ChainOfVerification
from src.utils import import_model_and_tokenizer


class ChainOfVerificationHuggingFace(ChainOfVerification):
    def __init__(
        self, model_id, top_p, temperature, task, setting, questions, hf_access_token
    ):
        super().__init__(model_id, task, setting, questions)
        self.hf_access_token = hf_access_token
        self.top_p = top_p
        self.temperature = temperature

        self.model, self.tokenizer = import_model_and_tokenizer(
            self.model_config, access_token=self.hf_access_token
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
            #if self.task_config.id != "multi_qa":
            tokens = tokens.split("[/INST]")[1]
        # TODO: Fix this for othe LLMs
        # else: 
        #     prompt = f"<|system|>\n</s>\n<|user|>\n{prompt_tmpl}</s>\n<|assistant|>\n"
        #     input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        #     outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=self.top_p,temperature=self.temperature)
        #     tokens = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt_tmpl):]
        #     tokens = tokens.split("<|assistant|>")[1]
        if len(tokens.split("\n\n")) > 1 and tokens.split("\n\n")[1] is not None:
            return tokens.split("\n\n")[1]
        else:
            return tokens

    def get_baseline_response(self, question: str) -> str:
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
            self.task_config.joint.plan_and_execute_prompt.format(
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
