import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import ChatOllama
from langchain_community.llms import Ollama
import os
## Langsmith Tracking
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# for var in ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"]:
#     value = os.getenv(var)
#     if value is not None:
#         os.environ[var] = value
st.title("Chat with Webpage 🌐")
st.caption("This app allows you to chat with a webpage using local deepseek-r1 and RAG")

# Get the webpage URL from the user
webpage_url = st.text_input("Enter Webpage URL", type="default")
# Connect to Ollama
# ollama_endpoint = "http://127.0.0.1:11434"
# model = "llama3.1:latest"
model = "deepseek-r1:8b"
ollama = Ollama(model=model, base_url="http://localhost:11434")

# Use session state to cache loaded data/vectorstore for the same URL
if 'last_url' not in st.session_state:
    st.session_state['last_url'] = None
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'docs' not in st.session_state:
    st.session_state['docs'] = None

if webpage_url:
    if st.session_state['last_url'] != webpage_url:
        with st.spinner("Loading webpage and building knowledge base..."):
            # 1. Load the data
            loader = WebBaseLoader(webpage_url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
            splits = text_splitter.split_documents(docs)

            # 2. Create Ollama embeddings and vector store
            embeddings = OllamaEmbeddings(model=model)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Store in session state
            st.session_state['last_url'] = webpage_url
            st.session_state['vectorstore'] = vectorstore
            st.session_state['retriever'] = retriever
            st.session_state['docs'] = docs
            st.success(f"Loaded {webpage_url} successfully!")
    else:
        vectorstore = st.session_state['vectorstore']
        retriever = st.session_state['retriever']
        docs = st.session_state['docs']
        st.info("URL unchanged. Using cached data.")

    # 3. Call Ollama Llama3 model
    def ollama_llm(question, context):
        """
        Generate a response to a question using the Ollama LLM with the provided context.
        Removes <think> tags from output if using deepseek model.
        Args:
            question (str): The user's question.
            context (str): The context to provide to the LLM.
        Returns:
            str: The LLM's answer as a string.
        """
        with st.spinner("Generating answer..."):
            formatted_prompt = f"Question: {question}\n\nContext: {context}"
            response = ollama.invoke([('human', formatted_prompt)])
            if isinstance(response, list) and response and "content" in response[0]:
                output = response[0]["content"].strip()
            else:
                output = str(response).strip()
            # If model is deepseek, remove <think> tags
            if model.startswith("deepseek"):
                import re
                output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
            return output

    def combine_docs(docs):
        """
        Combine the content of multiple document objects into a single string.
        Args:
            docs (list): List of document objects with a 'page_content' attribute.
        Returns:
            str: Combined content of all documents, separated by two newlines.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        """
        Retrieve relevant documents for a question and generate an answer using the LLM.
        Args:
            question (str): The user's question.
        Returns:
            str: The LLM's answer based on retrieved context.
        """
        with st.spinner("Retrieving relevant context..."):
            retrieved_docs = retriever.invoke(question)
            formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)


    # Ask a question about the webpage
    prompt = st.text_input("Ask any question about the webpage")

    # Chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)