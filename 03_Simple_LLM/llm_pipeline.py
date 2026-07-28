"""
LLM Pipeline Module
Runs a local HuggingFace FLAN-T5 model (default for hosted deploys where no
API key is available). Optional API providers are kept for higher-quality
inference when a key is configured.
"""

import os


class LLMAssistant:
    """Manages local FLAN-T5 inference with optional API upgrades."""

    DEFAULT_MODEL = "google/flan-t5-base"

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._init_local_model()

    def _init_local_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError(
                "transformers/torch is required. Install with: "
                "pip install transformers torch"
            ) from e

        cache_dir = os.environ.get("HF_HOME")
        print(f"Loading local LLM model: {self.model_name}...")
        load_kwargs = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **load_kwargs)
        print("Local LLM model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        api_key: str = None,
        provider: str = "Local FLAN-T5 (no key)",
    ) -> str:
        if not prompt.strip():
            return "Please provide a non-empty prompt."

        if provider == "Local FLAN-T5 (no key)":
            return self._generate_local(prompt, max_new_tokens, temperature)
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
        return f"Error: Unknown provider '{provider}'."

    def _generate_local(self, prompt, max_new_tokens, temperature):
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
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
                "Create one at https://huggingface.co/settings/tokens and paste it "
                "in the sidebar, or set HF_TOKEN in Streamlit secrets."
            )
        try:
            import requests
        except Exception:
            return "Error: 'requests' is not installed."

        model_id = os.environ.get("HF_MODEL_ID", self.DEFAULT_MODEL)
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
            if response.status_code == 503:
                return "HF Inference API Error 503: Model is loading. Retry in ~20s."
            return f"HF Inference API Error (HTTP {response.status_code}): {response.text[:500]}"
        except Exception as e:
            return f"HF Inference API Exception: {e}"

    def _call_gemini_api(self, prompt, api_key):
        try:
            import requests
        except Exception:
            return "Error: 'requests' is not installed."
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-pro:generateContent?key={api_key}"
            )
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"Gemini API Error (HTTP {response.status_code}): {response.text[:500]}"
        except Exception as e:
            return f"Gemini Request Exception: {e}"

    def _call_openai_api(self, prompt, api_key, max_new_tokens, temperature):
        try:
            import requests
        except Exception:
            return "Error: 'requests' is not installed."
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
