import io
import streamlit as st
from genai_engine import GenAIEngine

st.set_page_config(
    page_title="AuraGen: Creative GenAI Multimodal Canvas",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
    .genai-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #b92b27, #1565C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="genai-header">AuraGen: Creative Multimodal GenAI Studio</div>', unsafe_allow_html=True)
st.write("Synthesize AI Narrative Storyboards & High-Resolution Concept Artwork in real-time.")

@st.cache_resource
def get_engine():
    return GenAIEngine()

engine = get_engine()

col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("1. Concept & Style Prompt")
    concept_input = st.text_input("Enter Creative Theme / Concept:", value="Cybernetic Solitude in Lost Atlantis")
    style_preset = st.selectbox("Select Visual Aesthetic Style", [
        "Cyberpunk Neon",
        "Fantasy Mystic",
        "Sci-Fi Nebula",
        "Impressionist Sunset",
        "Minimalist Architectural"
    ])
    
    st.subheader("2. Generation Controls")
    resolution = st.select_slider("Art Canvas Resolution", options=[256, 512, 768], value=512)
    
    generate_btn = st.button("✨ Synthesize Story & Artwork", type="primary", use_container_width=True)

if generate_btn and concept_input:
    with st.spinner("Synthesizing Multimodal Narrative & Visual Art..."):
        storyboard = engine.generate_storyboard(concept_input, style_preset)
        art_img = engine.generate_concept_art(concept_input, style_preset, width=resolution, height=resolution)
        
        with col_out:
            st.subheader(f"📖 {storyboard['title']}")
            st.write(f"**Narrative:** {storyboard['narrative']}")
            st.code(f"Visual Prompt: {storyboard['visual_prompt']}", language="markdown")
            
            st.subheader("🖼️ Generated Concept Artwork")
            st.image(art_img, caption=f"{style_preset} - {resolution}x{resolution} px", use_container_width=True)
            
            # Download Image Button
            buf = io.BytesIO()
            art_img.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Concept Art (PNG)",
                data=buf.getvalue(),
                file_name="auragen_concept_art.png",
                mime="image/png"
            )
