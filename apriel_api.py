from keys import HF_TOKEN
from openai import OpenAI

class Apriel:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )
    
    def generate(self, prompt: str, max_tokens: int=16_384):
        completion = self.client.chat.completions.create(
            model="ServiceNow-AI/Apriel-1.6-15b-Thinker:together",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
        )
        response = completion.choices[0].message.content
        reasoning = completion.choices[0].message.model_extra['reasoning']
        return response, reasoning
