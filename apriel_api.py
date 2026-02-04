from typing import List, Union

from keys import HF_TOKEN
from openai import OpenAI

class Apriel:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )
    
    def generate(self, prompt: Union[str, List[str]], max_tokens: int=16_384):
        """Generate a response from the Apriel model.
        
        Args:
            prompt: Either a single string (user message) or a list of turns
                    for multi-turn conversation [user_turn, assistant_turn, user_turn2, ...].
                    Turns alternate between user and assistant roles.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Tuple of (response, reasoning).
        """
        if isinstance(prompt, str):
            # Single turn: wrap the string as a user message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        else:
            # Multi-turn: alternate between user and assistant
            messages = []
            for i, turn in enumerate(prompt):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": turn
                        }
                    ]
                })
        
        completion = self.client.chat.completions.create(
            model="ServiceNow-AI/Apriel-1.6-15b-Thinker:together",
            messages=messages,
            max_tokens=max_tokens,
        )
        response = completion.choices[0].message.content
        reasoning = completion.choices[0].message.model_extra['reasoning']
        return response, reasoning
