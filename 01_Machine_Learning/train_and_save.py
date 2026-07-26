"""
Solar Power Plant Energy Output & Efficiency ML Pipeline
Trains all standard scikit-learn regression and classification models, compares metrics,
generates performance plots, and saves the best models for deployment in Streamlit.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def generate_solar_dataset(n_samples=1500, seed=42):
    """Generates a realistic synthetic dataset for Solar Power Plant performance."""
    np.random.seed(seed)
    
    irradiance = np.random.uniform(200, 1100, n_samples)  # W/m2
    ambient_temp = np.random.uniform(10, 45, n_samples)   # Deg C
    module_temp = ambient_temp + (irradiance * 0.035) + np.random.normal(0, 2, n_samples)
    humidity = np.random.uniform(15, 90, n_samples)       # %
    wind_speed = np.random.uniform(0.5, 12.0, n_samples)  # m/s
    panel_age = np.random.uniform(0.1, 15.0, n_samples)   # Years
    inverter_eff = np.random.uniform(88.0, 98.5, n_samples) # %
    maintenance_score = np.random.uniform(50, 100, n_samples) # Score 50-100
    
    locations = np.random.choice(['Desert', 'Coastal', 'Urban', 'Rural'], size=n_samples, p=[0.35, 0.25, 0.2, 0.2])
    
    # Feature Engineering inside dataset creation
    degradation = 1.0 - (panel_age * 0.0075) + (maintenance_score / 1000.0)
    temp_penalty = 1.0 - np.maximum(0, (module_temp - 25) * 0.004)
    
    # Target 1: Power Output in kW (Continuous Regression Target)
    base_power = (irradiance * 0.45) * (inverter_eff / 100.0) * degradation * temp_penalty
    noise = np.random.normal(0, 12.0, n_samples)
    power_output = np.clip(base_power + noise, 0, None)
    
    # Target 2: Efficiency Tier (Categorical Classification Target: 0=Low, 1=Medium, 2=High)
    eff_ratio = power_output / (irradiance * 0.45 + 1e-5)
    quantiles = np.quantile(eff_ratio, [0.33, 0.67])
    
    efficiency_tier = np.zeros(n_samples, dtype=int)
    efficiency_tier[eff_ratio >= quantiles[0]] = 1
    efficiency_tier[eff_ratio >= quantiles[1]] = 2
    
    # Introduce controlled missing values to demonstrate preprocessing
    missing_mask_temp = np.random.rand(n_samples) < 0.03
    ambient_temp[missing_mask_temp] = np.nan
    
    missing_mask_wind = np.random.rand(n_samples) < 0.02
    wind_speed[missing_mask_wind] = np.nan

    df = pd.DataFrame({
        'irradiance_wm2': irradiance,
        'ambient_temp_c': ambient_temp,
        'module_temp_c': module_temp,
        'humidity_pct': humidity,
        'wind_speed_m_s': wind_speed,
        'panel_age_years': panel_age,
        'inverter_eff_pct': inverter_eff,
        'maintenance_score': maintenance_score,
        'location_type': locations,
        'power_output_kw': power_output,
        'efficiency_tier': efficiency_tier
    })
    return df

def run_ml_pipeline():
    print("=== Step 1: Data Collection & Preparation ===")
    df = generate_solar_dataset(n_samples=1500)
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    df.to_csv(os.path.join(models_dir, 'solar_power_dataset.csv'), index=False)
    print(f"Dataset generated with shape: {df.shape}")

    print("\n=== Step 2: Feature Engineering & Preprocessing ===")
    df['temp_diff'] = df['module_temp_c'] - df['ambient_temp_c'].fillna(df['ambient_temp_c'].mean())
    df['eff_index'] = (df['irradiance_wm2'] * df['inverter_eff_pct']) / 100.0
    
    X = df.drop(columns=['power_output_kw', 'efficiency_tier'])
    y_reg = df['power_output_kw']
    y_cls = df['efficiency_tier']

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(ohe_feature_names)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_processed, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_processed, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    print("\n=== Step 3: Regression Model Comparisons ===")
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
    }

    reg_results = []
    best_reg_score = -float('inf')
    best_reg_model = None
    best_reg_name = ""

    for name, model in reg_models.items():
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_test_r)
        
        mse = mean_squared_error(y_test_r, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_r, preds)
        r2 = r2_score(y_test_r, preds)
        
        reg_results.append({'Model': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2})
        print(f"[{name}] R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        if r2 > best_reg_score:
            best_reg_score = r2
            best_reg_model = model
            best_reg_name = name

    reg_df = pd.DataFrame(reg_results).sort_values(by='R2 Score', ascending=False)
    print(f"\n>> Best Regression Model: {best_reg_name} (R2 = {best_reg_score:.4f})")

    print("\n=== Step 4: Classification Model Comparisons ===")
    cls_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gaussian NB': GaussianNB()
    }

    cls_results = []
    best_cls_score = -float('inf')
    best_cls_model = None
    best_cls_name = ""

    for name, model in cls_models.items():
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_test_c)
        
        acc = accuracy_score(y_test_c, preds)
        prec = precision_score(y_test_c, preds, average='weighted')
        rec = recall_score(y_test_c, preds, average='weighted')
        f1 = f1_score(y_test_c, preds, average='weighted')
        
        cls_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        print(f"[{name}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_cls_score:
            best_cls_score = f1
            best_cls_model = model
            best_cls_name = name

    cls_df = pd.DataFrame(cls_results).sort_values(by='F1 Score', ascending=False)
    print(f"\n>> Best Classification Model: {best_cls_name} (F1 = {best_cls_score:.4f})")

    print("\n=== Step 5: Saving Models & Artifacts ===")
    joblib.dump(best_reg_model, os.path.join(models_dir, 'solar_regressor.joblib'))
    joblib.dump(best_cls_model, os.path.join(models_dir, 'solar_classifier.joblib'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    joblib.dump(all_feature_names, os.path.join(models_dir, 'feature_names.joblib'))
    
    reg_df.to_csv(os.path.join(models_dir, 'regression_metrics.csv'), index=False)
    cls_df.to_csv(os.path.join(models_dir, 'classification_metrics.csv'), index=False)
    
    print("Models and metrics saved successfully to 'models/' folder!")

if __name__ == '__main__':
    run_ml_pipeline()
