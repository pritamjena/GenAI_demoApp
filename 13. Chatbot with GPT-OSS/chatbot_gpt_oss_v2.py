import streamlit as st
from langchain_community.llms import Ollama
import time

# Required for actual BLEU/ROUGE
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# --- Streamlit App Setup ---
st.set_page_config(page_title="Model Comparison Chatbot", page_icon="🤖")
st.title("🤖 Model Comparison: GPT-OSS vs DeepSeek (with ROUGE & BLEU Scores)")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper for Validation Score ---
def compute_scores(ans1, ans2):
    # Compute BLEU and ROUGE-L between the two model answers as proxies (since no gold reference)
    reference = [ans1.split()]  # treat ans1 as reference for ans2
    hypothesis = ans2.split()
    bleu = sentence_bleu(reference, hypothesis)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(ans1, ans2)['rougeL'].fmeasure
    return bleu, rouge

# --- Initialize LLMs ---
@st.cache_resource
def get_llm(model_name):
    return Ollama(model=model_name)

# --- Chat Interface ---
st.divider()
st.subheader("💬 Ask a Question")

user_input = st.chat_input("Ask me anything!")
if user_input:
    # Generate both model answers with timing
    def get_answer_with_timing(model_name):
        llm = get_llm(model_name)
        start_time = time.time()
        answer = llm.invoke(user_input)
        end_time = time.time()
        time_taken = end_time - start_time
        return answer, time_taken

    with st.spinner("Generating answers from both models..."):
        answer_oss, time_oss = get_answer_with_timing("gpt-oss:20b")
        answer_ds, time_ds = get_answer_with_timing("deepseek-r1:8b")
        bleu, rouge = compute_scores(answer_oss, answer_ds)

    # Update and display chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("GPT-OSS", answer_oss, time_oss))
    st.session_state.chat_history.append(("DeepSeek", answer_ds, time_ds))
    st.session_state.chat_history.append(("score", f"BLEU: {bleu:.2f}, ROUGE-L: {rouge:.2f}"))

# --- Render chat ---
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message}")
    elif role == "GPT-OSS":
        with st.chat_message("assistant"):
            st.markdown(f"**GPT-OSS Answer:** {message[0]}")
            st.markdown(f"⏱️ **Time taken:** {message[1]:.2f} seconds")
    elif role == "DeepSeek":
        with st.chat_message("assistant"):
            st.markdown(f"**DeepSeek Answer:** {message[0]}")
            st.markdown(f"⏱️ **Time taken:** {message[1]:.2f} seconds")
    elif role == "score":
        st.markdown(f"🟢 **Validation Score — {message}**")

# --- Clear chat button ---
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
