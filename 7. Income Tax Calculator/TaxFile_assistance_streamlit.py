import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import re
# Streamlit app title
st.title("🧾 Income Tax Assistant (with Ollama)")
st.write("Ask questions about income tax calculation in India. Powered by local vector DB and LLM.")

# URLs to build the knowledge base from
urls = [
    "https://tax2win.in/guide/how-to-calculate-income-tax-on-salary",
    "https://cleartax.in/paytax/taxcalculator",
]

@st.cache_resource(show_spinner=True)
def get_vectorstore():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )
    return vectorstore

# Initialize LLM (not cached, lightweight)
llm = OllamaLLM(model="deepseek-r1:8b")

# Get or build the vectorstore
with st.spinner("Loading or building vector database..."):
    vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

# Build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def ask_tax_assistant(question: str):
    """Query the tax assistant with a user question."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    
    output = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    sources = result.get("source_documents", [])
    return output, sources

# Streamlit UI for question input
st.write("---")
st.subheader("Ask a question about income tax:")
user_question = st.text_input("Your question", "How do I calculate tax on my salary?")

if st.button("Get Answer"):
    with st.spinner("Thinking..."):
        answer, sources = ask_tax_assistant(user_question)
        st.success("Answer:")
        st.write(answer)
        if sources:
            st.write("\n**Sources:**")
            for i, src in enumerate(sources, 1):
                page_content = src.page_content[:200] + ("..." if len(src.page_content) > 200 else "")
                st.caption(f"{i}. {page_content}")