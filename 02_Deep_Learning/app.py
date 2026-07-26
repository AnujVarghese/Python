import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from train_models import IndustrialVitalsANN, DefectCNN, generate_synthetic_defect_image

st.set_page_config(
    page_title="Deep Learning Intelligence: ANN & CNN",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .title-banner {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner">Industrial AI Diagnostic Suite: ANN & CNN</div>', unsafe_allow_html=True)
st.write("Artificial Neural Network (ANN) for Tabular Sensor Vitals & Convolutional Neural Network (CNN) for Computer Vision Surface Quality Control.")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_ann_model():
    ann_path = os.path.join(MODEL_DIR, 'ann_vitals_model.pth')
    scaler_path = os.path.join(MODEL_DIR, 'ann_scaler.joblib')
    
    if not (os.path.exists(ann_path) and os.path.exists(scaler_path)):
        from train_models import train_ann_model
        train_ann_model()
        
    model = IndustrialVitalsANN(input_dim=5, num_classes=3)
    model.load_state_dict(torch.load(ann_path, map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_resource
def load_cnn_model():
    cnn_path = os.path.join(MODEL_DIR, 'cnn_defect_model.pth')
    if not os.path.exists(cnn_path):
        from train_models import train_cnn_model
        train_cnn_model()
        
    model = DefectCNN(num_classes=4)
    model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    ann_model, ann_scaler = load_ann_model()
    cnn_model = load_cnn_model()
    models_ready = True
except Exception as e:
    st.error(f"Error initializing models: {e}")
    models_ready = False

if models_ready:
    tab1, tab2 = st.tabs(["📊 ANN: Sensor Vitals Diagnostics", "🖼️ CNN: Surface Defect Inspection"])
    
    with tab1:
        st.subheader("Tabular Industrial Equipment Failure Predictor (ANN)")
        st.write("Input real-time sensor metrics to evaluate machine operational risk.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            vibration = st.slider("Vibration Level (mm/s)", 0.1, 5.0, 1.2, 0.1)
            temperature = st.slider("Operating Temperature (°C)", 30.0, 110.0, 55.0, 1.0)
        with c2:
            pressure = st.slider("System Pressure (Bar)", 1.0, 10.0, 4.5, 0.1)
            rpm = st.slider("Rotational Speed (RPM)", 800, 3500, 1800, 50)
        with c3:
            noise = st.slider("Acoustic Noise (dB)", 40.0, 100.0, 65.0, 1.0)
            
        if st.button("🔮 Diagnose Sensor Vitals", type="primary"):
            input_feats = np.array([[vibration, temperature, pressure, rpm, noise]])
            scaled_feats = ann_scaler.transform(input_feats)
            tensor_feats = torch.FloatTensor(scaled_feats)
            
            with torch.no_grad():
                logits = ann_model(tensor_feats)
                probs = F.softmax(logits, dim=1).numpy()[0]
                pred_cls = np.argmax(probs)
                
            labels = ["Normal Operational State 🟢", "Minor Warning / Maintenance Needed 🟡", "Critical System Failure Risk 🔴"]
            st.markdown(f"### Diagnosis: {labels[pred_cls]}")
            
            prob_df = pd.DataFrame({
                'Class': ['Normal', 'Warning', 'Critical'],
                'Probability': probs
            })
            fig, ax = plt.subplots(figsize=(6, 2.5))
            sns.barplot(data=prob_df, x='Probability', y='Class', palette='crest', ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)
            
    with tab2:
        st.subheader("Visual Quality Control & Defect Detection (CNN)")
        st.write("Upload a surface sample image or generate a sample test defect.")
        
        option = st.radio("Image Source", ["Generate Synthetic Defect Sample", "Upload Image File"], horizontal=True)
        
        img = None
        if option == "Generate Synthetic Defect Sample":
            defect_type = st.selectbox("Select Defect Type to Generate", ["Smooth (Normal)", "Scratch", "Crack", "Stain"])
            type_map = {"Smooth (Normal)": 0, "Scratch": 1, "Crack": 2, "Stain": 3}
            img = generate_synthetic_defect_image(type_map[defect_type])
        else:
            uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert('RGB')
                
        if img is not None:
            col_img, col_pred = st.columns(2)
            with col_img:
                st.image(img, caption="Input Surface Image (64x64)", width=220)
                
            with col_pred:
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = cnn_model(img_tensor)
                    probs = F.softmax(outputs, dim=1).numpy()[0]
                    pred_idx = np.argmax(probs)
                    
                classes = ["Smooth (Normal)", "Scratch Defect", "Crack Defect", "Stain Defect"]
                st.markdown(f"### Identified Class: **{classes[pred_idx]}**")
                
                df_probs = pd.DataFrame({
                    'Defect Category': classes,
                    'Confidence': probs
                })
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(data=df_probs, x='Confidence', y='Defect Category', palette='viridis', ax=ax)
                ax.set_xlim(0, 1)
                st.pyplot(fig)
