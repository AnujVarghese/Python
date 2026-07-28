# Project 3: CodeIQ & DocSense (Simple LLM Assistant)

A minimal yet practical Large Language Model application for automated code auditing, refactoring, and document intelligence.

---

## 📌 Features
- **Code Audit & Refactoring**: Detect potential bugs, optimize algorithms, and generate unit tests.
- **Executive Summarizer**: Summarize technical documents, articles, and reports into key actionable insights.
- **Prompt Playground**: Interactively adjust temperature, max tokens, and custom system prompts.
- **No API key required**: Default backend runs `google/flan-t5-base` locally — works offline and on Streamlit Cloud.
- **Optional API upgrades**: Swap to HuggingFace Inference API, Google Gemini, or OpenAI from the sidebar for higher quality.

---

## 📂 Folder Structure
```
03_Simple_LLM/
├── llm_pipeline.py              # Local FLAN-T5 inference + optional API backends
├── app.py                       # Streamlit Web UI
├── requirements.txt             # streamlit, torch, transformers
├── .streamlit/secrets.toml.example
└── README.md
```

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, `flan-t5-base` (~250 MB) downloads into the HuggingFace cache. After that, generation runs fully offline.

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo.
2. On <https://share.streamlit.io>, create a new app pointing at `03_Simple_LLM/app.py`.
3. Deploy — **no secrets required**. The app starts on `Local FLAN-T5 (no key)`.

### Memory note
Streamlit Cloud's free tier caps memory at ~1 GB. `flan-t5-base` + `torch` + `transformers` is tight but usually fits. If you hit OOM, add this to a `packages.txt` at the repo root to skip the heavy system libs:
```
libomp
```

### Optional: switch to a hosted backend
Add API keys in **App settings → Secrets** to upgrade quality:
```toml
HF_TOKEN = "hf_..."
GOOGLE_API_KEY = "..."
OPENAI_API_KEY = "sk-..."
```
Then pick the matching provider in the sidebar.