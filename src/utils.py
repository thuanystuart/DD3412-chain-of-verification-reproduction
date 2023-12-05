import dataclasses
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

@dataclasses.dataclass
class Model():
    id: str
    is_llama: bool = False
    is_protected: bool = False

MODEL_MAPPING = {
    "mistral": Model(id="mistralai/Mistral-7B-Instruct-v0.1", is_llama=False, is_protected=False),
    "llama2": Model(id="meta-llama/Llama-2-13b-chat-hf", is_llama=True, is_protected=True),
    "zephyr": Model(id="HuggingFaceH4/zephyr-7b-beta", is_llama=False, is_protected=False),
    "llama2_70b": Model(id="meta-llama/Llama-2-70b-chat-hf", is_llama=True, is_protected=True),
    "llama": Model(id="huggyllama/llama-65b", is_llama=True, is_protected=False),
}


def import_model_and_tokenizer(model: Model, access_token: str = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model.is_protected and access_token is not None:
        login(token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model.id,
            quantization_config=bnb_config,
            use_cache=True,
            device_map="auto",
            token=access_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model.id, quantization_config=bnb_config, use_cache=True, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model.id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
