# Project 3: CodeIQ & DocSense (Simple LLM Assistant)

A minimal yet practical Large Language Model application for automated code auditing, refactoring, and document intelligence.

---

## 📌 Features
- **Code Audit & Refactoring**: Detect potential bugs, optimize algorithms, and generate unit tests.
- **Executive Summarizer**: Summarize technical documents, articles, and reports into key actionable insights.
- **Prompt Playground**: Interactively adjust temperature, max tokens, and custom system prompts.
- **Hosted by default**: Uses the **HuggingFace Inference API** (free tier) on Streamlit Cloud — no large model download, no torch/transformers required at deploy time.
- **Optional local mode**: Run fully offline with `google/flan-t5-base` by uncommenting `transformers` + `torch` in `requirements.txt`.

---

## 📂 Folder Structure
```
03_Simple_LLM/
├── llm_pipeline.py              # Inference engine (HF Inference API + Gemini + OpenAI + local fallback)
├── app.py                       # Streamlit Web UI
├── requirements.txt             # Lightweight deps for hosted deploy
├── .streamlit/secrets.toml.example
└── README.md
```

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
# optional offline mode:
# pip install transformers torch
streamlit run app.py
```

Default backend is the **HuggingFace Inference API** — paste a free token from <https://huggingface.co/settings/tokens> in the sidebar (or set `HF_TOKEN` in `.streamlit/secrets.toml`).

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo.
2. On <https://share.streamlit.io>, create a new app pointing at `03_Simple_LLM/app.py`.
3. In **App settings → Secrets**, paste:
   ```toml
   HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"
   ```
   (and/or `GOOGLE_API_KEY`, `OPENAI_API_KEY` if you want those backends).
4. Deploy. `torch`/`transformers` are intentionally NOT installed, keeping the build under the 1 GB memory limit.

The first HF Inference call may take ~20s while the model warms up — `wait_for_model=true` is set, so it will auto-retry.
---
