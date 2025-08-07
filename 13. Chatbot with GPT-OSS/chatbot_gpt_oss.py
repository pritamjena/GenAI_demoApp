import streamlit as st
from langchain_community.llms import Ollama

# --- Streamlit App ---
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")
st.title("🤖 Simple Q&A Chatbot with Ollama")

# --- Session State for Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Initialize LLM ---
@st.cache_resource
def get_llm():
    return Ollama(model="gpt-oss:20b")

llm = get_llm()

st.divider()
st.subheader("💬 Chat")

# --- Chat input ---
user_input = st.chat_input("Ask me anything!")
if user_input:
    # Get answer from LLM
    with st.spinner("Thinking..."):
        answer = llm.invoke(user_input)

    # Update chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", answer))

# --- Render chat ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# --- Clear chat button ---
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
