# Project 3: CodeIQ & DocSense (Simple LLM Assistant)

A minimal yet practical Large Language Model application for automated code auditing, refactoring, and document intelligence.

---

## 📌 Features
- **Code Audit & Refactoring**: Detect potential bugs, optimize algorithms, and generate unit tests.
- **Executive Summarizer**: Summarize technical documents, articles, and reports into key actionable insights.
- **Prompt Playground**: Interactively adjust temperature, max tokens, and custom system prompts.
- **Hybrid Backend**: Runs locally offline using HuggingFace Transformers (`google/flan-t5-small`) out-of-the-box, with optional API support for Google Gemini or OpenAI.

---

## 📂 Folder Structure
```
03_Simple_LLM/
├── llm_pipeline.py    # LLM inference engine & API handlers
├── app.py             # Streamlit Web UI
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```
---
