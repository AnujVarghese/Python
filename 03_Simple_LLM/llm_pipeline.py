"""
LLM Pipeline Module
Handles text processing, local HuggingFace LLM inference (FLAN-T5 base/small),
and optional API key routing for Gemini or OpenAI.
"""

import os
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class LLMAssistant:
    """Manages LLM text generation, code auditing, and document summarization."""
    
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.generator = None
        self._init_local_model()

    def _init_local_model(self):
        try:
            print(f"Loading local LLM model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.generator = True
            print("Local LLM model loaded successfully!")
        except Exception as e:
            print(f"Failed to load HuggingFace model '{self.model_name}': {e}")
            self.generator = None
            self.tokenizer = None
            self.model = None

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, api_key: str = None, provider: str = "Local HuggingFace") -> str:
        """Executes text generation using chosen provider or local fallback."""
        if not prompt.strip():
            return "Please provide a non-empty prompt."
            
        # Optional API Provider: Gemini or OpenAI if selected and key provided
        if provider == "Google Gemini API" and api_key:
            return self._call_gemini_api(prompt, api_key)
        elif provider == "OpenAI API" and api_key:
            return self._call_openai_api(prompt, api_key)

        # Default Local Inference
        if self.generator is not None and self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                do_sample = temperature > 0.1
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = max(0.1, temperature)
                outputs = self.model.generate(**inputs, **gen_kwargs)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"Local Generation Error: {e}"
        else:
            return "Error: Local LLM pipeline is not available."

    def _call_gemini_api(self, prompt: str, api_key: str) -> str:
        """Calls Google Gemini API."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            payload = {'contents': [{'parts': [{'text': prompt}]}]}
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Gemini API Error (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"Gemini Request Exception: {e}"

    def _call_openai_api(self, prompt: str, api_key: str) -> str:
        """Calls OpenAI API via REST."""
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512
            }
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"OpenAI API Error (HTTP {response.status_code}): {response.text}"
        except Exception as e:
            return f"OpenAI Request Exception: {e}"
