import streamlit as st
import logging
from web_assist import fetch_url_content, extract_links, answer_query
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
for var in ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]:
    value = os.getenv(var)
    if value is not None:
        os.environ[var] = value
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="web_assist_streamlit.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def build_vector_db(start_url: str, output_file: str = "vector_store.faiss"):
    logging.info(f"Fetching links from: {start_url}")
    all_urls = [start_url] + extract_links(start_url)
    logging.info(f"Found {len(all_urls)} URLs to process")
    all_content = []
    for i, url in enumerate(all_urls):
        logging.info(f"Processing URL {i+1}/{len(all_urls)}: {url}")
        content = fetch_url_content(url)
        if content:
            all_content.append(content)
    full_text = "\n\n".join(all_content)
    logging.info(f"Total content size: {len(full_text)} characters")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    logging.info(f"Created {len(chunks)} text chunks")
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(output_file)
    logging.info(f"Vector database saved to {output_file}")
    return output_file

st.title("🔗 Website Q&A with Ollama")
st.write("Enter a website URL to build a knowledge base, then ask questions!")

url = st.text_input("Website URL", "https://enps.nsdl.com/eNPS/NationalPensionSystem.html")
build_db = st.button("Build Knowledge Base")

if build_db:
    # Only build DB if URL has changed
    last_url = st.session_state.get("last_url", None)
    if last_url != url:
        with st.spinner("Building vector database..."):
            try:
                db_path = build_vector_db(url)
                st.success(f"Vector database built for {url}!")
                st.session_state["db_path"] = db_path
                st.session_state["last_url"] = url
            except Exception as e:
                st.error(f"Error building vector DB: {e}")
                logging.error(f"Error building vector DB: {e}")
    else:
        st.info("URL unchanged. Using existing vector database.")

if "db_path" in st.session_state:
    st.write("---")
    st.subheader("Ask a question about the website:")
    user_query = st.text_input("Your question", "What is the National Pension System?")
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            try:
                answer = answer_query(user_query, db_path=st.session_state["db_path"])
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error answering question: {e}")
                logging.error(f"Error answering question: {e}") 