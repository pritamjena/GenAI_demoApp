import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import traceback
from typing import Sequence, Optional
from langchain_core.callbacks.manager import Callbacks
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self
import json


# Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "bge-m3:latest"
LLM_MODEL = "deepseek-r1:1.5b"
FAISS_INDEX_PATH = "vector_store.faiss"
TOP_K_RETRIEVAL = 15
RERANK_TOP_N = 5
PDF_PATH = "bhagavad-gita-in-english-source-file.pdf"
FLASHRANK_CACHE_DIR = os.path.expanduser("~/.cache/flashrank")

# Q&A Types
QA_TYPES = {
    "General Question": "Ask any general question about the Bhagavad Gita",
    "Philosophical Inquiry": "Deep philosophical questions about dharma, karma, and spirituality",
    "Verse Explanation": "Get explanations of specific verses or concepts",
    "Practical Guidance": "Seek practical advice based on Gita teachings",
    "Character Analysis": "Learn about characters like Arjuna, Krishna, and their roles",
    "Conceptual Understanding": "Understand key concepts like moksha, yoga, and devotion"
}

# Custom FlashRank Compressor
class FlashRankCompressor(BaseDocumentCompressor):
    model_name: str = Field(default="ms-marco-MiniLM-L-12-v2", description="FlashRank model name")
    top_n: int = Field(default=5, description="Number of top documents to return")
    cache_dir: str = Field(default="./flashrank_cache", description="Cache directory for models")
    _ranker: Optional[Ranker] = PrivateAttr(default=None)

    @model_validator(mode='after')
    def initialize_ranker(self) -> Self:
        self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        return self

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:
            return []
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)
        top_results = results[:self.top_n]
        final_results = []
        for r in top_results:
            metadata = r["meta"].copy()
            metadata["relevance_score"] = r["score"]
            doc = Document(
                page_content=r["text"],
                metadata=metadata
            )
            final_results.append(doc)
        return final_results


def setup_chain():
    """Initialize and setup the RAG chain with vector store and models."""
    print("Loading documents...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully!")
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("FAISS index created and saved!")
    
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )
    
    compressor = FlashRankCompressor(
        model_name="ms-marco-MiniLM-L-12-v2",
        top_n=RERANK_TOP_N,
        cache_dir=FLASHRANK_CACHE_DIR
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever, llm, vectorstore


def create_prompt_template(qa_type: str) -> PromptTemplate:
    """Create a prompt template based on the Q&A type."""
    base_template = """You are a wise sage drawing exclusively from the Bhagavad Gita.

Use ONLY the following context from the Gita to answer the question.
Do not add external knowledge or opinions.
If the context doesn't address the query, say "This query is not covered in the provided Gita verses."
When possible, cite the verse number(s) from the context.

Context:
{context}

Question: {question}

Response (strictly Gita-based):"""
    
    # Add specific guidance based on Q&A type
    if qa_type == "Philosophical Inquiry":
        base_template = base_template.replace(
            "Response (strictly Gita-based):",
            "Response (strictly Gita-based, focusing on philosophical depth):"
        )
    elif qa_type == "Verse Explanation":
        base_template = base_template.replace(
            "Response (strictly Gita-based):",
            "Response (strictly Gita-based, with detailed verse analysis):"
        )
    elif qa_type == "Practical Guidance":
        base_template = base_template.replace(
            "Response (strictly Gita-based):",
            "Response (strictly Gita-based, with practical application):"
        )
    elif qa_type == "Character Analysis":
        base_template = base_template.replace(
            "Response (strictly Gita-based):",
            "Response (strictly Gita-based, focusing on character insights):"
        )
    elif qa_type == "Conceptual Understanding":
        base_template = base_template.replace(
            "Response (strictly Gita-based):",
            "Response (strictly Gita-based, with clear conceptual explanation):"
        )
    
    return PromptTemplate(
        template=base_template,
        input_variables=["context", "question"]
    )


def format_docs(docs):
    """Format retrieved documents with metadata."""
    return "\n\n".join([
        f"[Verse {i+1}]\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])


def ask_gita(question: str, qa_type: str, compression_retriever, llm) -> dict:
    """
    Query the Gita with source document tracking and Q&A type-specific processing.
    
    Args:
        question: User's question
        qa_type: Type of Q&A (from QA_TYPES)
        compression_retriever: The retriever for document search
        llm: The language model
        
    Returns:
        Dictionary with response and source documents
    """
    try:
        # Get compressed documents for transparency
        compressed_docs = compression_retriever.invoke(question)
        
        # Create type-specific prompt
        prompt_template = create_prompt_template(qa_type)
        
        # Build RAG chain with type-specific prompt
        rag_chain = (
            {
                "context": compression_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Generate response
        response = rag_chain.invoke(question)
        
        # Extract source information
        sources = [
            {
                "content": doc.page_content[:150] + "...",
                "page": doc.metadata.get("page", "Unknown"),
                "relevance_score": doc.metadata.get("relevance_score", "N/A")
            }
            for doc in compressed_docs[:3]  # Top 3 sources
        ]
        
        return {
            "response": response,
            "sources": sources,
            "num_sources": len(compressed_docs),
            "qa_type": qa_type
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


def extract_after_think(response: str) -> str:
    """Extract response after thinking tags from DeepSeek model."""
    marker = "</think>"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].lstrip()
    return response


def setup_streamlit_ui():
    """Setup the Streamlit user interface."""
    st.set_page_config(
        page_title="Bhagavad Gita Chatbot",
        page_icon="🕉️",
        layout="wide"
    )
    
    st.title("🕉️ Bhagavad Gita Chatbot")
    st.markdown("Ask questions about the Bhagavad Gita and receive answers based exclusively on its teachings.")
    
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This chatbot uses **Retrieval Augmented Generation (RAG)** to answer questions 
            based on the Bhagavad Gita.
            
            **Features:**
            - 🔍 Semantic search with BGE-M3 embeddings
            - 🎯 FlashRank reranking for improved relevance
            - 🤖 DeepSeek-R1 8B for response generation
            - 📄 Source attribution with relevance scores
            - 🎭 Q&A type-specific responses
            """
        )
        
        st.header("Configuration")
        st.markdown(f"**Embedding Model:** {EMBEDDING_MODEL}")
        st.markdown(f"**LLM Model:** {LLM_MODEL}")
        st.markdown(f"**Chunk Size:** {CHUNK_SIZE}")
        st.markdown(f"**Top-K Retrieval:** {TOP_K_RETRIEVAL}")
        st.markdown(f"**Rerank Top-N:** {RERANK_TOP_N}")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main function to run the Bhagavad Gita chatbot."""
    # Setup Streamlit UI
    setup_streamlit_ui()
    
    # Initialize the RAG chain
    with st.spinner("Initializing Gita chatbot..."):
        compression_retriever, llm, vectorstore = setup_chain()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Q&A Type Selection
    st.subheader("Select Question Type")
    qa_type = st.selectbox(
        "Choose the type of question you want to ask:",
        options=list(QA_TYPES.keys()),
        help="Select the type of question to get more targeted responses"
    )
    st.info(f"**{qa_type}**: {QA_TYPES[qa_type]}")
    
    # Question Input
    user_question = st.text_area(
        "Ask the Gita your question...",
        placeholder="Enter your question here...",
        height=100
    )
    
    # Submit Button
    if st.button("Ask Question", type="primary"):
        if user_question.strip():
            with st.spinner("Thinking..."):
                result = ask_gita(user_question, qa_type, compression_retriever, llm)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display response
                st.subheader("Response")
                st.write(extract_after_think(result["response"]))
                
                # Display Q&A type info
                st.info(f"**Question Type:** {result['qa_type']}")
                
                # Display sources
                st.subheader(f"Relevant Sources (Top {len(result['sources'])})")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i} (Page {source['page']}, Relevance: {source['relevance_score']:.3f})"):
                        st.code(source["content"])
                
                # Add to chat history
                st.session_state.messages.append({
                    "question": user_question,
                    "response": result["response"],
                    "qa_type": qa_type,
                    "sources": result["sources"]
                })
        else:
            st.warning("Please enter a question.")
    
    # Display Chat History
    if st.session_state.messages:
        st.subheader("Chat History")
        for i, message in enumerate(reversed(st.session_state.messages[-5:])):  # Show last 5
            with st.expander(f"Q: {message['question'][:50]}... ({message['qa_type']})"):
                st.write("**Question:**", message['question'])
                st.write("**Response:**", extract_after_think(message['response']))
                st.write("**Type:**", message['qa_type'])
                if st.button(f"Delete", key=f"delete_{i}"):
                    st.session_state.messages.pop(-(i+1))
                    st.rerun()


if __name__ == "__main__":
    main()
