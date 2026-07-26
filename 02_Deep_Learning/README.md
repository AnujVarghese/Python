# Project 2: Industrial AI Diagnostic Suite (ANN & CNN)

A comprehensive Deep Learning solution engineered using **PyTorch** for tabular sensor telemetry classification (ANN) and surface quality control computer vision (CNN).

---

## 📌 Features

### 1. Artificial Neural Network (ANN) - Sensor Diagnostics
- **Input**: 5 tabular sensor features (Vibration, Operating Temp, Pressure, RPM, Noise Level).
- **Architecture**: 3 Dense Layers with `BatchNorm1d`, `ReLU`, and `Dropout(0.2)`.
- **Target**: Machine Operational Risk (0: Normal, 1: Warning, 2: Critical).

### 2. Convolutional Neural Network (CNN) - Quality Inspection
- **Input**: 64x64 RGB Surface Images.
- **Augmentation Pipeline**: `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`, `Normalize`.
- **Architecture**: 3 Convolutional blocks (`Conv2d`, `BatchNorm2d`, `MaxPool2d`, `ReLU`) + Dense Classifier.
- **Target**: Surface Quality (Smooth, Scratch, Crack, Stain).

---

## 📂 Folder Structure
```
02_Deep_Learning/
├── dl_ann_cnn.ipynb     # PyTorch Notebook with training execution & loss/accuracy curves
├── train_models.py      # Standalone PyTorch dataset generator & model trainer
├── app.py               # Dual-tab Streamlit Web Application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── models/              # Saved PyTorch (.pth) & Scaler (.joblib) models
```

---

## 🚀 Execution Guide

### 1. Train Models
```bash
python train_models.py
```
Or run `dl_ann_cnn.ipynb` in Jupyter Notebook.

### 2. Launch Streamlit UI
```bash
streamlit run app.py
```
---
