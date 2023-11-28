import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import time 


import json

def read_json_file(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records

file_path = '/proj/layegh/users/x_amila/CoV/wikidata_questions.json'
data = read_json_file(file_path)


llama_model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
zephyr_model_id = "HuggingFaceH4/zephyr-7b-beta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(zephyr_model_id,
                                             quantization_config=bnb_config,
                                             use_cache=True,
                                             device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(zephyr_model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

BASELINE_PROMPT_WIKI = """Answer the below question which is asking for a list of persons. Output should be a numbered list and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Question: {original_question}

Answer:"""


VERIFICATION_QUESTION_PROMPT_WIKI = """Your task is to create a series of verification questions based on the below question, the verfication question template and baseline response.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Was [movie actor] born in Boston?
Example Baseline Response: 1. Matt Damon 
2. Chris Evans 
Verification questions: 1. Where was Matt Damon born?
2. Where was Chirs Evans born?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {original_question}
Baseline Response: {baseline_response}
Verification Questions:"""

EXECUTE_PLAN_PROMPT_SEARCH_TOOL = """Answer the following questions. Think step by step and answer each question concisely.

Questions: {verification_questions}

Answers:"""

FINAL_VERIFIED_PROMPT = """Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer.
Original Query: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs:
{verification_questions} & {verification_answers}

Final Refined Answer:"""

questions = []
for record in data:
    for i in range (0, len(record)):
        question = list(record.keys())[i]
        questions.append(question)
        

for q in questions[:30]:
    baseline_prompt_tmpl = BASELINE_PROMPT_WIKI.format(original_question=q)
    prompt = f"<|system|>\n</s>\n<|user|>\n{baseline_prompt_tmpl}</s>\n<|assistant|>\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    baseline_outputs = model.generate(input_ids=input_ids, max_new_tokens=80, do_sample=True, top_p=0.9,temperature=0.7)
    baseline_tokens = tokenizer.batch_decode(baseline_outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(baseline_prompt_tmpl):]
    baseline_tokens = baseline_tokens.split("<|assistant|>")[1]
    
    verification_question_prompt_tmpl = VERIFICATION_QUESTION_PROMPT_WIKI.format(original_question=q, baseline_response=baseline_tokens)
    verif_prompt = f"<|system|>\n</s>\n<|user|>\n{verification_question_prompt_tmpl}</s>\n<|assistant|>\n"
    verif_input_ids = tokenizer(verif_prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    verif_outputs = model.generate(input_ids=verif_input_ids, max_new_tokens=90, do_sample=True, top_p=0.9,temperature=0.7)
    verif_tokens = tokenizer.batch_decode(verif_outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(verification_question_prompt_tmpl):]
    verif_tokens = verif_tokens.split("<|assistant|>")[1]
    
    execute_questions_prompt_tmpl = EXECUTE_PLAN_PROMPT_SEARCH_TOOL.format(verification_questions=verif_tokens)
    execute_verif_prompt = f"<|system|>\n</s>\n<|user|>\n{execute_questions_prompt_tmpl}</s>\n<|assistant|>\n"
    execute_verif_input_ids = tokenizer(execute_verif_prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    execute_verif_outputs = model.generate(input_ids=execute_verif_input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.7)
    execute_verif_tokens = tokenizer.batch_decode(execute_verif_outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(execute_questions_prompt_tmpl):]
    execute_verif_tokens = execute_verif_tokens.split("<|assistant|>")[1]
    
    final_verified_prompt_tmpl = FINAL_VERIFIED_PROMPT.format(original_question=q, baseline_response=baseline_tokens, verification_questions=verif_tokens, verification_answers=execute_verif_tokens)
    final_verified_prompt = f"<|system|>\n</s>\n<|user|>\n{final_verified_prompt_tmpl}</s>\n<|assistant|>\n"
    final_verified_input_ids = tokenizer(final_verified_prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    final_verified_outputs = model.generate(input_ids=final_verified_input_ids, max_new_tokens=80, do_sample=True, top_p=0.9,temperature=0.7)
    final_verified_tokens = tokenizer.batch_decode(final_verified_outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(final_verified_prompt_tmpl):]
    final_verified_tokens = final_verified_tokens.split("<|assistant|>")[1]
    
    
    
    print(f"Question: {q}\n")
    print(f"Baseline Answer: {baseline_tokens}\n")
    print(f"Verification Questions: {verif_tokens}\n")
    print(f"Execute Plan: {execute_verif_tokens}\n")
    print(f"Final Refined Answer: {final_verified_tokens}\n")
    print("---------------------------------------------------\n")
