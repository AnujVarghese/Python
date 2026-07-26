# 🚀 Antigravity 6 Complete AI & ML Projects Suite

A comprehensive collection of 6 end-to-end, production-grade Artificial Intelligence and Machine Learning applications built using **Python**, **Jupyter Notebooks**, **PyTorch**, **Scikit-Learn**, **OpenCV**, **Transformers**, and **Streamlit**.

---

## 📚 Portfolio Projects Index

| # | Domain | Project Name & Description | Tech Stack & Models | Web UI |
|---|--------|----------------------------|---------------------|--------|
| 1 | **Machine Learning** | **Solar Power Plant Energy & Efficiency Suite**: Dual continuous regression (power output in kW) & categorical classification (efficiency tier: Low, Med, High). | Scikit-Learn (11 Regressors, 9 Classifiers), Pandas, Seaborn, Joblib | `01_Machine_Learning/app.py` |
| 2 | **Deep Learning** | **Industrial AI Diagnostic Suite**: Dual PyTorch ANN (tabular machine sensor vitals) & CNN (2D surface defect visual inspection). | PyTorch, Torchvision, Convolutional & Dense Nets | `02_Deep_Learning/app.py` |
| 3 | **Simple LLM** | **CodeIQ & DocSense**: AI Code Reviewer, Refactoring Assistant, & Executive Document Summarizer. | Transformers (`FLAN-T5`), Optional Gemini/OpenAI API | `03_Simple_LLM/app.py` |
| 4 | **OpenCV Vision** | **SmartVision Studio**: Automated Document Edge Scanner & Perspective Warper, Color HSV Segmentation, Face Blur Privacy Filter, Digital Filter Studio. | OpenCV (`cv2`), NumPy, PIL | `04_OpenCV_Vision/app.py` |
| 5 | **Generative AI** | **AuraGen Creative Canvas**: Multimodal Narrative Storyboard & Procedural Abstract Visual Concept Art Synthesizer. | Transformers, PIL Canvas Engine, NumPy | `05_Generative_AI/app.py` |
| 6 | **AI Chatbot** | **NexusChat AI**: Context-aware conversational assistant with customizable personas, token metrics, and chat history export. | Transformers, Streamlit Session State | `06_AI_Chatbot/app.py` |

---

## 📁 Repository Directory Structure

```
D:\Fake/
├── 01_Machine_Learning/
│   ├── solar_power_ml.ipynb   # Jupyter notebook with EDA, preprocessing & model comparisons
│   ├── train_and_save.py       # ML pipeline script
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt
│   └── README.md
│
├── 02_Deep_Learning/
│   ├── dl_ann_cnn.ipynb       # PyTorch ANN & CNN training notebook
│   ├── train_models.py        # PyTorch model trainer script
│   ├── app.py                 # Streamlit UI (Sensor Vitals & Surface Defect inspection)
│   ├── requirements.txt
│   └── README.md
│
├── 03_Simple_LLM/
│   ├── llm_pipeline.py        # LLM local & API inference pipeline
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt
│   └── README.md
│
├── 04_OpenCV_Vision/
│   ├── vision_processor.py    # OpenCV image processing engine
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt
│   └── README.md
│
├── 05_Generative_AI/
│   ├── genai_engine.py        # GenAI storyboard & visual art engine
│   ├── app.py                 # Streamlit UI
│   ├── requirements.txt
│   └── README.md
│
└── 06_AI_Chatbot/
    ├── chatbot_engine.py      # Chatbot persona & context memory engine
    ├── app.py                 # Streamlit UI
    ├── requirements.txt
    └── README.md
```

---

## ⚡ Quick Start Guide

To run any project's Streamlit interface:

```bash
# Navigate to project folder
cd 01_Machine_Learning  # Or 02_Deep_Learning, 03_Simple_LLM, etc.

# Install dependencies
pip install -r requirements.txt

# Run Streamlit UI
streamlit run app.py
```
---
