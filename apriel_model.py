import re
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "./Apriel-1.6-15b-Thinker"

class Apriel:
    def __init__(self, path=MODEL_PATH):
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model, self.processor = model, processor

    def generate(self, prompt: str, max_tokens: int=16_384):
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.6)

        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        try:
            response = re.findall(r"\[BEGIN FINAL RESPONSE\](.*?)(?:<\|end\|>)", output, re.DOTALL)[0].strip()
        except:
            response = None

        return response, output
