"""
LLM Pipeline Module
Handles remote inference via HuggingFace Inference API, Google Gemini, or OpenAI.
Local transformers/torch fallback is optional and only loaded if those packages
are already installed (keeps Streamlit Cloud installs light).
"""

import os
import requests


def _try_import_transformers():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        return AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        return None, None


class LLMAssistant:
    """Manages LLM text generation, code auditing, and document summarization."""

    DEFAULT_HF_MODEL = "google/flan-t5-base"

    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        self.model = None
        self._init_local_model()

    def _init_local_model(self):
        AutoTokenizer, AutoModelForSeq2SeqLM = _try_import_transformers()
        if AutoTokenizer is None:
            print("transformers/torch not installed — skipping local model load.")
            return
        try:
            print(f"Loading local LLM model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.generator = True
            print("Local LLM model loaded successfully!")
        except Exception as e:
            print(f"Failed to load local model '{self.model_name}': {e}")
            self.generator = None
            self.tokenizer = None
            self.model = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        api_key: str = None,
        provider: str = "HuggingFace Inference API",
    ) -> str:
        if not prompt.strip():
            return "Please provide a non-empty prompt."

        if provider == "HuggingFace Inference API":
            return self._call_hf_inference(prompt, api_key, max_new_tokens, temperature)
        if provider == "Google Gemini API":
            if not api_key:
                return "Error: Google Gemini API selected but no API key provided."
            return self._call_gemini_api(prompt, api_key)
        if provider == "OpenAI API":
            if not api_key:
                return "Error: OpenAI API selected but no API key provided."
            return self._call_openai_api(prompt, api_key, max_new_tokens, temperature)
        if provider == "Local HuggingFace (FLAN-T5)":
            if self.model is not None and self.tokenizer is not None:
                return self._generate_local(prompt, max_new_tokens, temperature)
            return (
                "Error: Local LLM unavailable on this host (torch/transformers "
                "not installed or model failed to load). Pick a hosted backend."
            )

        return f"Error: Unknown provider '{provider}'."

    def _generate_local(self, prompt, max_new_tokens, temperature):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            do_sample = temperature > 0.1
            gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = max(0.1, temperature)
            outputs = self.model.generate(**inputs, **gen_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Local Generation Error: {e}"

    def _call_hf_inference(self, prompt, api_key, max_new_tokens, temperature):
        if not api_key:
            return (
                "Error: HuggingFace Inference API requires an API token. "
                "Create a free one at https://huggingface.co/settings/tokens and "
                "paste it in the sidebar, or set HF_TOKEN in Streamlit secrets."
            )
        model_id = os.environ.get("HF_MODEL_ID", self.DEFAULT_HF_MODEL)
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": max(0.1, temperature),
                "do_sample": temperature > 0.1,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                return str(data)
            if response.status_code == 401:
                return "HF Inference API Error 401: Invalid or missing token."
            if response.status_code == 403:
                return f"HF Inference API Error 403: Access denied for model '{model_id}'."
            if response.status_code == 503:
                return (
                    "HF Inference API Error 503: Model is loading. "
                    "Retry in ~20 seconds (wait_for_model is enabled)."
                )
            return f"HF Inference API Error (HTTP {response.status_code}): {response.text[:500]}"
        except requests.exceptions.Timeout:
            return "HF Inference API Error: Request timed out (60s)."
        except Exception as e:
            return f"HF Inference API Exception: {e}"

    def _call_gemini_api(self, prompt, api_key):
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-pro:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            return f"Gemini API Error (HTTP {response.status_code}): {response.text[:500]}"
        except Exception as e:
            return f"Gemini Request Exception: {e}"

    def _call_openai_api(self, prompt, api_key, max_new_tokens, temperature):
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(max_new_tokens, 1024),
                "temperature": max(0.0, temperature),
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return f"OpenAI API Error (HTTP {response.status_code}): {response.text[:500]}"
        except Exception as e:
            return f"OpenAI Request Exception: {e}"
