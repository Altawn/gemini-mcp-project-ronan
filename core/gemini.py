import google.generativeai as genai
from google.generativeai.types import ContentType

class Gemini:
    def __init__(self, model: str, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def add_user_message(self, messages: list, message):
        # Gemini utilise "user" et "model" comme rôles
        messages.append({"role": "user", "parts": [str(message)]})

    def add_assistant_message(self, messages: list, message):
        messages.append({"role": "model", "parts": [str(message)]})

    def text_from_message(self, message):
        # Extraction du texte de la réponse Gemini
        return message.text

    def chat(self, messages, system=None, temperature=1.0, stop_sequences=[], tools=None, thinking=False, thinking_budget=1024):
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            
            if "parts" in msg:
                parts = msg["parts"]
            else:
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle list content (e.g. text blocks)
                    parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                else:
                    parts = [str(content)]
            
            gemini_messages.append({"role": role, "parts": parts})

        # Configuration de la génération
        config = genai.types.GenerationConfig(
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        
        response = self.model.generate_content(
            gemini_messages,
            generation_config=config
        )
        
        # Monkey patch stop_reason for compatibility with Chat class
        # Gemini finish_reason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        # We default to "end_turn" which corresponds to standard completion
        response.stop_reason = "end_turn"
        
        return response