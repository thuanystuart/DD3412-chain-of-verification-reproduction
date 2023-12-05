import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login


def import_model_and_tokenizer(model_name, access_token=None):
    model_mapping = {
        "mistral": {"id": "mistralai/Mistral-7B-Instruct-v0.1", "is_llama": False},
        "llama2": {"id": "meta-llama/Llama-2-13b-chat-hf", "is_llama": True},
        "zephyr": {"id": "HuggingFaceH4/zephyr-7b-beta", "is_llama": False},
        "llama2_70b": {"id": "meta-llama/Llama-2-70b-chat-hf", "is_llama": True},
        "llama": {"id": "huggyllama/llama-65b", "is_llama": False},
    }

    model = model_mapping.get(model_name, None)
    if model is None:
        print("Invalid Model ID. Please write either 'mistral', 'llama2' or 'zephyr'.")
        sys.exit()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model.is_llama and access_token is not None:
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
