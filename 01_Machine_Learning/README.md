# Project 1: Solar Power Plant Energy Output & Efficiency ML Suite

A complete machine learning project featuring both **Regression** (Power Output Prediction in kW) and **Classification** (Efficiency Tier Classification) using standard Scikit-Learn algorithms, extensive feature engineering, preprocessing pipelines, and a Streamlit dashboard.

---

## 📌 Project Overview
- **Dataset**: Real-world operational metrics of solar power facilities including solar irradiance, ambient temperature, module temperature, humidity, wind speed, panel age, inverter efficiency, maintenance rating, and geographic location environment.
- **Regression Task**: Predict continuous power output (`power_output_kw`) across 11 Scikit-Learn models.
- **Classification Task**: Classify solar efficiency tier (`efficiency_tier`: 0=Low, 1=Medium, 2=High) across 9 Scikit-Learn models.
- **Preprocessing**: Imputation (median/most_frequent), One-Hot Encoding, StandardScaler, and engineered features (`temp_diff`, `eff_index`).

---

## 📂 Folder Structure
```
01_Machine_Learning/
├── solar_power_ml.ipynb   # Jupyter Notebook with full EDA, preprocessing & model comparisons
├── train_and_save.py       # Standalone training script to build & export models
├── app.py                 # Interactive Streamlit Web Application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── models/                # Saved artifacts (.joblib models, CSV benchmarks)
```

---

## 🛠️ Models Evaluated

### Regression Models (11 Algorithms):
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree Regressor
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. AdaBoost Regressor
9. Extra Trees Regressor
10. Support Vector Regressor (SVR)
11. K-Neighbors Regressor

### Classification Models (9 Algorithms):
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. AdaBoost Classifier
6. Extra Trees Classifier
7. Support Vector Classifier (SVC)
8. K-Neighbors Classifier
9. Gaussian Naive Bayes

---

## 🚀 How to Run

### 1. Train Models & Generate Artifacts
```bash
python train_and_save.py
```
Or open and execute `solar_power_ml.ipynb` in Jupyter Notebook.

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```
---
