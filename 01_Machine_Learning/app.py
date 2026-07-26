import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Solar Power ML Intelligence Suite",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff8c00, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_artifacts():
    reg_path = os.path.join(MODEL_DIR, 'solar_regressor.joblib')
    cls_path = os.path.join(MODEL_DIR, 'solar_classifier.joblib')
    prep_path = os.path.join(MODEL_DIR, 'preprocessor.joblib')
    
    if not (os.path.exists(reg_path) and os.path.exists(cls_path) and os.path.exists(prep_path)):
        # Auto train if missing
        from train_and_save import run_ml_pipeline
        run_ml_pipeline()
        
    reg_model = joblib.load(reg_path)
    cls_model = joblib.load(cls_path)
    preprocessor = joblib.load(prep_path)
    
    reg_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'regression_metrics.csv'))
    cls_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'classification_metrics.csv'))
    
    return reg_model, cls_model, preprocessor, reg_metrics, cls_metrics

try:
    reg_model, cls_model, preprocessor, reg_metrics, cls_metrics = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    artifacts_loaded = False

st.markdown('<div class="main-header">Solar Power Plant Energy & Efficiency AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dual Machine Learning Architecture: Power Output Regression & Efficiency Tier Classification</div>', unsafe_allow_html=True)

if artifacts_loaded:
    tab1, tab2, tab3 = st.tabs(["🚀 Real-Time Inference", "📊 Model Comparison & Leaderboard", "📈 Exploratory Data Analytics"])
    
    with tab1:
        st.subheader("Predict Plant Performance")
        st.write("Adjust operational parameters in the sidebar or below to simulate solar plant behavior.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            irradiance = st.slider("Solar Irradiance (W/m²)", 200.0, 1200.0, 850.0, 10.0)
            ambient_temp = st.slider("Ambient Temperature (°C)", 5.0, 50.0, 28.0, 0.5)
            module_temp = st.slider("Module Temperature (°C)", 10.0, 75.0, 48.0, 0.5)
        with col2:
            humidity = st.slider("Relative Humidity (%)", 10.0, 95.0, 45.0, 1.0)
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 4.2, 0.1)
            panel_age = st.slider("Panel Age (Years)", 0.1, 20.0, 3.5, 0.1)
        with col3:
            inverter_eff = st.slider("Inverter Efficiency (%)", 85.0, 99.0, 96.5, 0.1)
            maintenance_score = st.slider("Maintenance Rating (50-100)", 50.0, 100.0, 88.0, 1.0)
            location_type = st.selectbox("Location Environment", ["Desert", "Coastal", "Urban", "Rural"])
            
        # Engineer input dataframe
        temp_diff = module_temp - ambient_temp
        eff_index = (irradiance * inverter_eff) / 100.0
        
        input_data = pd.DataFrame([{
            'irradiance_wm2': irradiance,
            'ambient_temp_c': ambient_temp,
            'module_temp_c': module_temp,
            'humidity_pct': humidity,
            'wind_speed_m_s': wind_speed,
            'panel_age_years': panel_age,
            'inverter_eff_pct': inverter_eff,
            'maintenance_score': maintenance_score,
            'location_type': location_type,
            'temp_diff': temp_diff,
            'eff_index': eff_index
        }])
        
        if st.button("⚡ Run AI Diagnostics", type="primary", use_container_width=True):
            try:
                processed_input = preprocessor.transform(input_data)
                
                # Predictions
                pred_power = reg_model.predict(processed_input)[0]
                pred_tier_idx = cls_model.predict(processed_input)[0]
                pred_probs = cls_model.predict_proba(processed_input)[0] if hasattr(cls_model, 'predict_proba') else [0.33, 0.33, 0.34]
                
                tier_names = {0: "Low Efficiency 🔴", 1: "Medium Efficiency 🟡", 2: "High Efficiency 🟢"}
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Predicted Power Output", f"{pred_power:.2f} kW")
                with res_col2:
                    st.metric("Predicted Efficiency Tier", tier_names[pred_tier_idx])
                    
                st.subheader("Tier Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Tier': ['Low (0)', 'Medium (1)', 'High (2)'],
                    'Probability': pred_probs
                })
                fig, ax = plt.subplots(figsize=(6, 2.5))
                sns.barplot(data=prob_df, x='Probability', y='Tier', palette='mako', ax=ax)
                ax.set_xlim(0, 1)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Inference error: {e}")
                
    with tab2:
        st.subheader("Model Evaluation & Leaderboard")
        col_reg, col_cls = st.columns(2)
        
        with col_reg:
            st.markdown("### 📉 Regression Models Benchmark")
            st.dataframe(reg_metrics.style.highlight_max(axis=0, subset=['R2 Score'], color='darkgreen'))
            
            fig_r, ax_r = plt.subplots(figsize=(6, 4))
            sns.barplot(data=reg_metrics, x='R2 Score', y='Model', palette='Blues_r', ax=ax_r)
            ax_r.set_title("R2 Score Comparison")
            st.pyplot(fig_r)
            
        with col_cls:
            st.markdown("### 🏷️ Classification Models Benchmark")
            st.dataframe(cls_metrics.style.highlight_max(axis=0, subset=['F1 Score'], color='darkgreen'))
            
            fig_c, ax_c = plt.subplots(figsize=(6, 4))
            sns.barplot(data=cls_metrics, x='F1 Score', y='Model', palette='Purples_r', ax=ax_c)
            ax_c.set_title("Weighted F1 Score Comparison")
            st.pyplot(fig_c)
            
    with tab3:
        st.subheader("Exploratory Dataset Insights")
        dataset_path = os.path.join(MODEL_DIR, 'solar_power_dataset.csv')
        if os.path.exists(dataset_path):
            df_raw = pd.read_csv(dataset_path)
            st.write(f"Sample Dataset ({df_raw.shape[0]} records):")
            st.dataframe(df_raw.head(10))
            
            fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
            num_df = df_raw.select_dtypes(include=[np.number])
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
            st.pyplot(fig_corr)
