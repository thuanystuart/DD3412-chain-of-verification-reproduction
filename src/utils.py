import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import sys

def import_model_and_tokenizer(model_id, access_token=None):
    mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    llama2_model_id = "meta-llama/Llama-2-13b-chat-hf"
    zephyr_model_id = "HuggingFaceH4/zephyr-7b-beta"
    llama2_70b_model_id = "meta-llama/Llama-2-70b-chat-hf"
    llama_model_id = "huggyllama/llama-65b"
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)
    if model_id == "mistral":
        model_id = mistral_model_id
    elif model_id == "llama2":
        model_id = llama2_model_id
    elif model_id == "zephyr":
        model_id = zephyr_model_id
    elif model_id == "llama2_70b":
        model_id = llama2_70b_model_id
    elif model_id == "llama-65b":
        model_id = llama_model_id
    else:
        print("Invalid Model ID. Please write either 'mistral', 'llama2' or 'zephyr'.")
        sys.exit()
    
    if model_id == llama2_model_id:
        login(token = access_token)
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 use_cache=True,
                                                 device_map='auto',
                                                 token=access_token
                                                 )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                 use_cache=True,
                                                 device_map='auto',)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer