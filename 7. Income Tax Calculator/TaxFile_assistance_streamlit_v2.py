import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import re
from langchain_core.messages import SystemMessage, HumanMessage
import torch
from langchain_ollama import OllamaLLM

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
        chunk_size=500, 
        chunk_overlap=50
    )
    
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Use local HuggingFace embeddings instead of Ollama
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
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
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

def simple_rerank(query, documents, top_n=3):
    """Simple reranking based on keyword matching"""
    query_words = set(query.lower().split())
    
    scored_docs = []
    for doc in documents:
        doc_words = set(doc.page_content.lower().split())
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(doc_words))
        union = len(query_words.union(doc_words))
        score = intersection / union if union > 0 else 0
        
        scored_docs.append({
            'document': doc.page_content,
            'relevance_score': score,
            'metadata': getattr(doc, 'metadata', {})
        })
    
    # Sort by relevance score and return top_n
    scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_docs[:top_n]




system_prompt = (
    "You are an expert Indian Income Tax assistant. "
    "Provide clear, accurate, and up-to-date answers on tax rules, slabs, exemptions, deductions, "
    "filing procedures, and recent regulatory changes as per Indian law. "
    "Always cite legal sections or official sources when possible. "
    "If you are not sure, suggest users consult an official tax consultant or the Income Tax Department."
)


def ask_tax_assistant(question: str):
    """Query the tax assistant with a user question."""
    retrieved = retriever.get_relevant_documents(question)
    # Apply simple reranking
    reranked = simple_rerank(question, retrieved, top_n=3)

    combined_context = "\n\n".join([d['document'] for d in reranked])
    # Form chat prompt with context
    # Create prompt for the LLM
    full_prompt = f"""
        {system_prompt}

        Context Information:
        {combined_context}

        Question: {question}

        Answer based on the provided context:
        """
    chat_llm = OllamaLLM(model="deepseek-r1:8b")
    response = chat_llm.invoke([('human', full_prompt)])
    output = str(response).strip()
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    return output, reranked

# Streamlit UI for question input
st.write("---")
st.subheader("Ask a question about income tax:")
user_question = st.text_input("Your question", "How do I calculate tax on my salary?")

if st.button("Get Answer") and user_question:
    with st.spinner("Processing your question..."):
        answer, sources = ask_tax_assistant(user_question)
        
        st.success("Answer:")
        st.write(answer)
        
        if sources:
            st.write("**Sources:**")
            for i, src in enumerate(sources, 1):
                with st.expander(f"Source {i} (Relevance: {src['relevance_score']:.3f})"):
                    st.write(src['document'][:500] + ("..." if len(src['document']) > 500 else ""))

# Add sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This tax assistant runs completely offline using:\n"
    "- Local HuggingFace embeddings\n"
    "- Deepseek 8b model\n"
    "- FAISS vector database\n\n"
    "No API calls are made to external services."
)

st.sidebar.warning(
    "**Disclaimer:** This tool provides general information only. "
    "Always consult with a qualified tax professional for specific advice.")