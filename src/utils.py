import dataclasses
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from src.prompts import (
    BASELINE_PROMPT_WIKI,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI,
    FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI,
    ##
    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI,
    FINAL_VERIFIED_JOINT_PROMPT_WIKI,
    ##
    EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI,
    ##
    BASELINE_PROMPT_WIKI_CATEGORY,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
    FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY,
    ##
    PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI_CATEGORY,
    FINAL_VERIFIED_JOINT_PROMPT_WIKI_CATEGORY,
    ##
    EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI_CATEGORY,
    ##
    BASELINE_PROMPT_MULTI_QA,
    PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
    EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
    FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA,
    ##
    PLAN_AND_EXECUTION_JOINT_PROMPT_MULTI_QA,
    FINAL_VERIFIED_JOINT_PROMPT_MULTI_QA,
    ##
    EXECUTE_VERIFICATION_FACTORED_PROMPT_MULTI_QA,
)

SETTINGS = ["two_step", "joint", "factored"]

@dataclasses.dataclass
class FactoredConfig:
    max_tokens_plan: int
    max_tokens_execute: int
    max_tokens_verify: int
    plan_prompt: str
    execute_prompt: str
    verify_prompt: str
    plan_command: str = " Verification Questions: "
    execute_command: str = " Answer: "
    verify_command: str = " Final Refined Answer: "

@dataclasses.dataclass
class TwoStepConfig:
    max_tokens_plan: int
    max_tokens_execute: int
    max_tokens_verify: int
    plan_prompt: str
    execute_prompt: str
    verify_prompt: str
    plan_command: str = " Verification Questions: "
    execute_command: str = " Answers: "
    verify_command: str = " Final Refined Answer: "

@dataclasses.dataclass
class JointConfig:
    max_tokens_plan_and_execute: int
    max_tokens_verify: int
    plan_and_execute_prompt: str
    verify_prompt: str
    plan_and_execute_command: str = " Verification Questions and Answers: "
    verify_command: str = " Final Refined Answer: "

@dataclasses.dataclass
class TaskConfig:
    id: str
    max_tokens: int
    baseline_prompt: str
    two_step: TwoStepConfig
    joint: JointConfig
    factored: FactoredConfig
    baseline_command: str = " Answer: "


TASK_MAPPING = {
    "wikidata": TaskConfig(
        id="wikidata",
        max_tokens=150,
        baseline_prompt=BASELINE_PROMPT_WIKI,
        two_step=TwoStepConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI,
            execute_prompt=EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI,
            max_tokens_plan=300,
            max_tokens_execute=300,
            max_tokens_verify=300,
        ),
        joint=JointConfig(
            plan_and_execute_prompt=PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI,
            verify_prompt=FINAL_VERIFIED_JOINT_PROMPT_WIKI,
            max_tokens_plan_and_execute=500,
            max_tokens_verify=150,
        ),
        factored=FactoredConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI,
            execute_prompt=EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI,
            max_tokens_plan=300,
            max_tokens_execute=70,
            max_tokens_verify=300,
        ),
    ),
    "multispanqa": TaskConfig(
        id="multispanqa",
        max_tokens=200,
        baseline_prompt=BASELINE_PROMPT_MULTI_QA,
        two_step=TwoStepConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
            execute_prompt=EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA,
            max_tokens_plan=400,
            max_tokens_execute=400,
            max_tokens_verify=300,
        ),
        joint=JointConfig(
            plan_and_execute_prompt=PLAN_AND_EXECUTION_JOINT_PROMPT_MULTI_QA,
            verify_prompt=FINAL_VERIFIED_JOINT_PROMPT_MULTI_QA,
            max_tokens_plan_and_execute=600,
            max_tokens_verify=200,
        ),
        factored=FactoredConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA,
            execute_prompt=EXECUTE_VERIFICATION_FACTORED_PROMPT_MULTI_QA,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA,
            max_tokens_plan=400,
            max_tokens_execute=90,
            max_tokens_verify=300,
        ),
    ),
    "wikidata_category": TaskConfig(
        id="wikidata_category",
        max_tokens=100,
        baseline_prompt=BASELINE_PROMPT_WIKI_CATEGORY,
        two_step=TwoStepConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
            execute_prompt=EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY,
            max_tokens_plan=300,
            max_tokens_execute=300,
            max_tokens_verify=150,
        ),
        joint=JointConfig(
            plan_and_execute_prompt=PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI_CATEGORY,
            verify_prompt=FINAL_VERIFIED_JOINT_PROMPT_WIKI_CATEGORY,
            max_tokens_plan_and_execute=400,
            max_tokens_verify=100,
        ),
        factored=FactoredConfig(
            plan_prompt=PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY,
            execute_prompt=EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI_CATEGORY,
            verify_prompt=FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY,
            max_tokens_plan=300,
            max_tokens_execute=70,
            max_tokens_verify=150,
        ),
    ),
}


@dataclasses.dataclass
class ModelConfig:
    id: str
    prompt_format: str
    is_llama: bool = False
    is_protected: bool = False

STD_PROMPT_FORMAT = """{prompt}"""
GPT_PROMPT_FORMAT = """{prompt}\n\nAnswer:"""
LLAMA_PROMPT_FORMAT = (
    """<s>[INST] <<SYS>>{prompt}\n<</SYS>>\n{command} [/INST]"""
)
MODEL_MAPPING = {
    "gpt3": ModelConfig(
        id="gpt-3.5-turbo-0613",
        prompt_format=GPT_PROMPT_FORMAT,
        is_llama=False,
        is_protected=False,
    ),
    "llama2": ModelConfig(
        id="meta-llama/Llama-2-13b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        is_llama=True,
        is_protected=True,
    ),
    "llama2_70b": ModelConfig(
        id="meta-llama/Llama-2-70b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        is_llama=True,
        is_protected=True,
    ),
    "llama-65b": ModelConfig(
        id="huggyllama/llama-65b",
        prompt_format=LLAMA_PROMPT_FORMAT,
        is_llama=True,
        is_protected=False,
    ),
}


def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(
        current_directory,
        rf'../{path_relative_to_project_root}'
    )

    return final_directory

def import_model_and_tokenizer(model: ModelConfig, access_token: str = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model.is_protected and access_token is not None:
        login(token=access_token)
        language_model = AutoModelForCausalLM.from_pretrained(
            model.id,
            quantization_config=bnb_config,
            use_cache=True,
            device_map="auto",
            token=access_token,
        )
    else:
        language_model = AutoModelForCausalLM.from_pretrained(
            model.id, quantization_config=bnb_config, use_cache=True, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model.id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return language_model, tokenizer
