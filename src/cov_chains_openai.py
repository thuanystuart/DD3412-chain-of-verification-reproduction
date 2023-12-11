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
        raise NotImplementedError("Subclasses must implement this method.")

    def get_baseline_response(self, question: str) -> str:
        baseline_response_prompt_template = PromptTemplate(
            input_variables=["original_question"],
            template=self.task_config.baseline_prompt
        )
        baseline_response_prompt_chain = \
            {"original_question": RunnablePassthrough()} | \
            baseline_response_prompt_template | \
            self.llm | \
            StrOutputParser()
        
        baseline_response = baseline_response_prompt_chain.invoke(question)
        return baseline_response

    def run_two_step_chain(self, question: str, baseline_response: str):
       raise NotImplementedError("Subclasses must implement this method.")

    def run_joint_chain(self, question: str, baseline_response: str):
        raise NotImplementedError("Subclasses must implement this method.")

