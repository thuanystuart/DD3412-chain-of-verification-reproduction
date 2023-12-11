from src.cove_chains import ChainOfVerification
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

    def call_llm(self, prompt: str, max_tokens: int) -> str:
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
            tokens = tokens.split("[/INST]")[1]
        
        if len(tokens.split("\n\n")) > 1 and tokens.split("\n\n")[1] is not None:
            return tokens.split("\n\n")[1]
        else:
            return tokens

    def process_prompt(self, prompt, command) -> str:
        return self.model_config.prompt_format.format(prompt=prompt, command=command)
