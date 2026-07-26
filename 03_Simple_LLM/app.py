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
st.write("Intelligent Code Review, Refactoring, and Executive Document Summarization powered by local Transformers & optional API routing.")

@st.cache_resource
def get_llm():
    return LLMAssistant()

llm = get_llm()

# Sidebar Configuration
st.sidebar.title("⚙️ LLM Configuration")
provider = st.sidebar.selectbox("Inference Backend", ["Local HuggingFace (FLAN-T5)", "Google Gemini API", "OpenAI API"])
api_key = ""
if provider in ["Google Gemini API", "OpenAI API"]:
    api_key = st.sidebar.text_input(f"Enter {provider} Key", type="password")
    
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 64, 512, 256, 32)

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
