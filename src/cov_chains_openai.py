from src.cov_chains import ChainOfVerification
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ChainOfVerificationOpenAI(ChainOfVerification):
    def __init__(
        self, model_id, temperature, task, setting, questions, openai_access_token
    ):
        super().__init__(model_id, task, setting)
        self.openai_access_token = openai_access_token
        self.questions = questions
        self.temperature = temperature
        
        self.llm = ChatOpenAI(
            openai_api_key=openai_access_token,
            model_name=self.model_config.id,
            max_tokens=500
        )

    def generate_response(self, prompt: str, max_tokens: int) -> str:
        llm_chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
        return llm_chain.invoke({})

    def get_baseline_response(self, question: str) -> str:
        baseline_prompt = self.task_config.baseline_prompt.format(
            original_question=question
        )
        return self.generate_response(baseline_prompt, self.task_config.max_tokens)

    def run_two_step_chain(self, question: str, baseline_response: str):
        # Create Plan
        plan_prompt = self.task_config.two_step.plan_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
        )
        plan_response = self.generate_response(plan_prompt, self.task_config.two_step.max_tokens_plan)
        
        ## Execute Plan
        execute_prompt = self.task_config.two_step.execute_prompt.format(
            verification_questions=plan_response
        )
        execute_response = self.generate_response(
            execute_prompt, self.task_config.two_step.max_tokens_execute
        )
        
        ## Verify
        verify_prompt = self.task_config.two_step.verify_prompt.format(
            original_question=question,
            baseline_response=baseline_response,
            verification_questions=plan_response,
            verification_answers=execute_response,
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
        raise NotImplementedError("Subclasses must implement this method.")

