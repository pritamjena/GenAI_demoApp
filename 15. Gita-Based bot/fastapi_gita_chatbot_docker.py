# fastapi_gita_chatbot.py

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Sequence
from contextlib import asynccontextmanager
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
from langchain_core.callbacks.manager import Callbacks
from pydantic import PrivateAttr, model_validator
from typing_extensions import Self
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
OLLAMA_EMBED_BASE_URL = os.getenv("OLLAMA_EMBED_BASE_URL", "http://ollama-embed:11434")
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://ollama-llm:11434")


# Q&A Types
QA_TYPES = {
    "General Question": "Ask any general question about the Bhagavad Gita",
    "Philosophical Inquiry": "Deep philosophical questions about dharma, karma, and spirituality",
    "Verse Explanation": "Get explanations of specific verses or concepts",
    "Practical Guidance": "Seek practical advice based on Gita teachings",
    "Character Analysis": "Learn about characters like Arjuna, Krishna, and their roles",
    "Conceptual Understanding": "Understand key concepts like moksha, yoga, and devotion"
}

# Pydantic Models for Request and Response
class QuestionRequest(BaseModel):
    """Request model for Gita questions"""
    question: str = Field(..., description="The question to ask the Bhagavad Gita", min_length=5)
    qa_type: str = Field(..., description="Type of question from available Q&A types")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the meaning of dharma according to Krishna?",
                "qa_type": "Philosophical Inquiry"
            }
        }

class SourceInfo(BaseModel):
    """Model for source information"""
    content: str = Field(..., description="Preview of the source content")
    page: str = Field(..., description="Page number from the PDF")
    relevance_score: float = Field(..., description="Relevance score from reranking")

class QuestionResponse(BaseModel):
    """Response model for Gita answers"""
    response: str = Field(..., description="Answer based on Bhagavad Gita teachings")
    sources: List[SourceInfo] = Field(..., description="Top source documents used")
    num_sources: int = Field(..., description="Total number of sources retrieved")
    qa_type: str = Field(..., description="Type of question asked")
    cleaned_response: str = Field(..., description="Response with thinking tags removed")

class QATypesResponse(BaseModel):
    """Response model for available Q&A types"""
    qa_types: Dict[str, str] = Field(..., description="Dictionary of available Q&A types and descriptions")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


# Custom FlashRank Compressor
class FlashRankCompressor(BaseDocumentCompressor):
    """Custom document compressor using FlashRank for reranking"""
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
    logger.info("Loading documents...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_EMBED_BASE_URL
    )
    
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully!")
    else:
        logger.info("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        logger.info("FAISS index created and saved!")
    
    llm = OllamaLLM(
        model=LLM_MODEL, 
        temperature=0.1,
        base_url=OLLAMA_LLM_BASE_URL
    )
    
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


def extract_after_think(response: str) -> str:
    """Extract response after thinking tags from DeepSeek model."""
    marker = "</think>"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].lstrip()
    return response


async def ask_gita(question: str, qa_type: str, compression_retriever, llm) -> dict:
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
                "page": str(doc.metadata.get("page", "Unknown")),
                "relevance_score": float(doc.metadata.get("relevance_score", 0.0))
            }
            for doc in compressed_docs[:3]  # Top 3 sources
        ]
        
        return {
            "response": response,
            "sources": sources,
            "num_sources": len(compressed_docs),
            "qa_type": qa_type,
            "cleaned_response": extract_after_think(response)
        }
    except Exception as e:
        logger.error(f"Error in ask_gita: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


# Global variables for chain components
compression_retriever = None
llm = None
vectorstore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG chain on startup"""
    global compression_retriever, llm, vectorstore
    try:
        logger.info("Initializing Gita chatbot...")
        compression_retriever, llm, vectorstore = setup_chain()
        logger.info("Gita chatbot initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {traceback.format_exc()}")
        raise
    yield
    # Cleanup code can go here if needed


# Initialize FastAPI app
app = FastAPI(
    title="Bhagavad Gita Chatbot API",
    description="API for querying the Bhagavad Gita using RAG with FlashRank reranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint for health check"""
    return HealthResponse(
        status="healthy",
        message="Bhagavad Gita Chatbot API is running"
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if compression_retriever is None or llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or unavailable"
        )
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.get("/qa-types", response_model=QATypesResponse, tags=["Q&A Types"])
async def get_qa_types():
    """Get available Q&A types and their descriptions"""
    return QATypesResponse(qa_types=QA_TYPES)


@app.post("/ask", response_model=QuestionResponse, tags=["Gita Q&A"], status_code=status.HTTP_200_OK)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the Bhagavad Gita chatbot.
    
    - **question**: The question to ask (minimum 5 characters)
    - **qa_type**: Type of question (must be one of the available QA types)
    
    Returns the answer with source documents and relevance scores.
    """
    # Validate Q&A type
    if request.qa_type not in QA_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid qa_type. Must be one of: {list(QA_TYPES.keys())}"
        )
    
    # Check if chatbot is initialized
    if compression_retriever is None or llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot is still initializing. Please try again in a moment."
        )
    
    try:
        logger.info(f"Processing question: {request.question[:50]}... (Type: {request.qa_type})")
        result = await ask_gita(request.question, request.qa_type, compression_retriever, llm)
        
        return QuestionResponse(
            response=result["response"],
            sources=[SourceInfo(**source) for source in result["sources"]],
            num_sources=result["num_sources"],
            qa_type=result["qa_type"],
            cleaned_response=result["cleaned_response"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """Get current chatbot configuration"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_retrieval": TOP_K_RETRIEVAL,
        "rerank_top_n": RERANK_TOP_N,
        "pdf_path": PDF_PATH
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_gita_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
