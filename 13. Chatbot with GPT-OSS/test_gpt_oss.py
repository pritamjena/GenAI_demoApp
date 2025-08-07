import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os
import hashlib

# 1. Streamlit UI for PDF upload and question input
st.title('PDF Question Answering with Ollama Embeddings + Ollama LLM [GPT-OSS]')
pdf_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about your PDF:")

# Initialize session state for caching
if 'pdf_hash' not in st.session_state:
    st.session_state.pdf_hash = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

if pdf_file is not None:
    # Calculate hash of the uploaded file to check if it's the same
    file_content = pdf_file.getvalue()
    current_hash = hashlib.md5(file_content).hexdigest()
    
    # Check if the PDF has changed
    if st.session_state.pdf_hash != current_hash:
        st.info("Processing new PDF file...")
        
        # 2. Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # 3. Load and split PDF using the temporary file path
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load_and_split()
            
            # 4. Use Ollama Embeddings (using the same model as LLM for consistency)
            embedder = OllamaEmbeddings(model="deepseek-r1:8b")
            vectorstore = FAISS.from_documents(documents, embedder)
            
            # 5. Connect to Ollama LLM (ensure ollama and gpt-oss:20b are running)
            llm = Ollama(model="gpt-oss:20b")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=vectorstore.as_retriever()
            )
            
            # Cache the results in session state
            st.session_state.pdf_hash = current_hash
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = qa
            
            st.success("PDF processed successfully!")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    else:
        st.info("Using cached PDF data...")
        vectorstore = st.session_state.vectorstore
        qa = st.session_state.qa_chain
    
    # 6. Query on submit
    if question and 'qa_chain' in st.session_state and st.session_state.qa_chain is not None:
        result = qa({"query": question})
        st.write("**Answer:**", result["result"])
