import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from llm_pipeline import LLMAssistant

st.set_page_config(
    page_title="CodeIQ & DocSense: AI LLM Studio",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .llm-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="llm-title">CodeIQ & DocSense: Practical LLM Assistant</div>', unsafe_allow_html=True)
st.write("Intelligent Code Review, Refactoring, and Executive Document Summarization. Hosted on Streamlit Cloud via the HuggingFace Inference API — no heavy model download required.")

@st.cache_resource
def get_llm():
    return LLMAssistant()

llm = get_llm()

# Sidebar Configuration
st.sidebar.title("⚙️ LLM Configuration")
provider = st.sidebar.selectbox(
    "Inference Backend",
    ["HuggingFace Inference API", "Google Gemini API", "OpenAI API", "Local HuggingFace (FLAN-T5)"],
    help="On Streamlit Cloud, use a hosted backend. Local mode requires torch+transformers installed locally.",
)

# Resolve API key: sidebar input takes precedence, else fall back to Streamlit secrets
secret_map = {
    "HuggingFace Inference API": "HF_TOKEN",
    "Google Gemini API": "GOOGLE_API_KEY",
    "OpenAI API": "OPENAI_API_KEY",
}
sidebar_key = st.sidebar.text_input(
    f"Enter {provider} Key (or set {secret_map.get(provider, 'API_KEY')} in secrets)",
    type="password",
    value="",
    help="Free tokens: huggingface.co/settings/tokens · aistudio.google.com · platform.openai.com",
)
api_key = sidebar_key or st.secrets.get(secret_map.get(provider, ""), "")

if not api_key and provider != "Local HuggingFace (FLAN-T5)":
    st.sidebar.warning(f"No {provider} key detected. Add one in the field above or in Streamlit secrets.")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 64, 1024, 256, 32)

tab1, tab2, tab3 = st.tabs(["💻 Code Audit & Refactoring", "📄 Executive Document Summarizer", "playground Prompt Playground"])

with tab1:
    st.subheader("Automated Code Inspection & Refactoring")
    code_input = st.text_area("Paste Code Snippet:", height=200, value="""def calc_average(lst):
    total = 0
    for i in range(len(lst)):
        total += lst[i]
    return total / len(lst)""")
    
    task_type = st.selectbox("Select Analysis Task", [
        "Explain Code & Potential Bugs",
        "Refactor & Optimize Code",
        "Generate Docstrings & Type Annotations",
        "Write PyTest Unit Tests"
    ])
    
    if st.button("🚀 Analyze Code", type="primary"):
        with st.spinner("Analyzing code via LLM..."):
            prompt = f"Task: {task_type}.\nCode:\n{code_input}"
            output = llm.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, api_key=api_key, provider=provider)
            
            st.markdown("### AI Output:")
            st.code(output, language="python" if "code" in task_type.lower() else "markdown")

with tab2:
    st.subheader("Document Summarization & Insight Extraction")
    doc_text = st.text_area("Paste Document or Article Text:", height=200, value="""Machine learning (ML) is a field of study in artificial intelligence devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.""")
    
    summary_mode = st.radio("Output Format", ["Concise Executive Summary", "Key Actionable Bullet Points", "Technical Deep-Dive"], horizontal=True)
    
    if st.button("📑 Summarize Document", type="primary"):
        with st.spinner("Processing document..."):
            prompt = f"Format: {summary_mode}.\nSummarize the following text:\n{doc_text}"
            output = llm.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, api_key=api_key, provider=provider)
            st.markdown("### Executive Summary:")
            st.write(output)

with tab3:
    st.subheader("Custom Prompt Playground")
    custom_prompt = st.text_area("Enter any Prompt:", height=150, value="What are the key differences between Convolutional Neural Networks and Transformers?")
    
    if st.button("✨ Submit Prompt"):
        with st.spinner("Generating response..."):
            output = llm.generate(custom_prompt, max_new_tokens=max_tokens, temperature=temperature, api_key=api_key, provider=provider)
            st.markdown("### Response:")
            st.write(output)
