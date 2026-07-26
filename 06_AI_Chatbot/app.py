import json
import streamlit as st
from chatbot_engine import ChatbotEngine, PERSONA_PROMPTS

st.set_page_config(
    page_title="NexusChat AI: Contextual Chatbot",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
<style>
    .chat-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8E2DE2, #4A00E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-header">NexusChat AI: Contextual Conversational Assistant</div>', unsafe_allow_html=True)

@st.cache_resource
def get_bot():
    return ChatbotEngine()

bot = get_bot()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
st.sidebar.title("🤖 Chat Settings")
persona = st.sidebar.selectbox("Select AI Persona", list(PERSONA_PROMPTS.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Conversation Analytics")
total_turns = len(st.session_state.messages) // 2
st.sidebar.metric("Turn Count", total_turns)
total_words = sum(len(m["content"].split()) for m in st.session_state.messages)
st.sidebar.metric("Est. Word Count", total_words)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

if st.session_state.messages:
    chat_export = json.dumps(st.session_state.messages, indent=2)
    st.sidebar.download_button(
        label="📥 Export Chat History (JSON)",
        data=chat_export,
        file_name="nexus_chat_history.json",
        mime="application/json",
        use_container_width=True
    )

# Display Chat Messages
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Chat Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Generate Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NexusChat is thinking..."):
            response = bot.respond(user_input, st.session_state.messages, persona)
            st.write(response)
            
    # Append Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": response})
