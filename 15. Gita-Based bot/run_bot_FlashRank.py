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
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import traceback
from typing import Sequence, Optional
from langchain_core.callbacks.manager import Callbacks
from pydantic import Field, PrivateAttr


# Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "bge-m3:latest"
LLM_MODEL = "deepseek-r1:8b"
FAISS_INDEX_PATH = "vector_store.faiss"
TOP_K_RETRIEVAL = 15
RERANK_TOP_N = 5
PDF_PATH = "bhagavad-gita-in-english-source-file.pdf"
FLASHRANK_CACHE_DIR = os.path.expanduser("~/.cache/flashrank")


# Custom FlashRank Compressor
from langchain_core.documents.compressor import BaseDocumentCompressor
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from typing import Sequence, Optional
from langchain_core.callbacks.manager import Callbacks
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

# Custom FlashRank Compressor
class FlashRankCompressor(BaseDocumentCompressor):
    """Custom document compressor using FlashRank Ranker directly."""
    
    model_name: str = Field(default="ms-marco-MiniLM-L-12-v2", description="FlashRank model name")
    top_n: int = Field(default=5, description="Number of top documents to return")
    cache_dir: str = Field(default="./flashrank_cache", description="Cache directory for models")
    
    # Use PrivateAttr for non-serializable objects
    _ranker: Optional[Ranker] = PrivateAttr(default=None)
    
    @model_validator(mode='after')
    def initialize_ranker(self) -> Self:
        """Initialize the FlashRank Ranker after model validation."""
        self._ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        return self
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using FlashRank Ranker.
        
        Args:
            documents: A sequence of documents to compress.
            query: The query to use for reranking.
            callbacks: Optional callbacks to run during compression.
            
        Returns:
            A sequence of reranked documents.
        """
        if len(documents) == 0:
            return []
        
        # Prepare passages for FlashRank
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            }
            for i, doc in enumerate(documents)
        ]
        
        # Create rerank request and rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)
        
        # Get top N results
        top_results = results[:self.top_n]
        
        # Convert back to Document objects with relevance scores
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


# Load documents
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

# Initialize embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Load or create vector store
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# Initialize LLM
llm = OllamaLLM(model=LLM_MODEL, temperature=0.3)

# Create base retriever
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_RETRIEVAL}
)

print(f"base_retriever is {base_retriever}")

# Initialize custom FlashRank compressor
print("Loading FlashRank reranker...")
compressor = FlashRankCompressor(
    model_name="ms-marco-MiniLM-L-12-v2",
    top_n=RERANK_TOP_N,
    cache_dir=FLASHRANK_CACHE_DIR
)

# Create compression retriever with custom FlashRank compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Define strict Gita-only prompt
prompt_template = """You are a wise sage drawing exclusively from the Bhagavad Gita.

Use ONLY the following context from the Gita to answer the question.
Do not add external knowledge or opinions.
If the context doesn't address the query, say "This query is not covered in the provided Gita verses."
When possible, cite the verse number(s) from the context.

Context:
{context}

Question: {question}

Response (strictly Gita-based):"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Build LCEL chain
def format_docs(docs):
    """Format retrieved documents with metadata"""
    return "\n\n".join([
        f"[Verse {i+1}]\n{doc.page_content}" 
        for i, doc in enumerate(docs)
    ])

# LCEL chain with pipe operators
rag_chain = (
    {
        "context": compression_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query function with source tracking
def ask_gita(question: str) -> dict:
    """
    Query the Gita with source document tracking
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with response and source documents
    """
    try:
        # Get compressed documents for transparency
        compressed_docs = compression_retriever.invoke(question)
        
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
            "num_sources": len(compressed_docs)
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

def extract_after_think(response: str) -> str:
    """
    Extracts and returns the part of the response after '</think>'.
    If '</think>' is not found, returns the original response.
    """
    marker = "</think>"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].lstrip()
    return response

# Example usage
if __name__ == "__main__":
    while True:
        user_q = input("\nAsk the Gita (or type 'exit'): ")
        if user_q.lower() == "exit":
            break
        result = ask_gita(user_q)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n" + "="*60)
            print("RESPONSE:")
            print("="*60)
            # Only show output after </think>
            print(extract_after_think(result["response"]))
            print("\n" + "="*60)
            print(f"SOURCES (Retrieved {result['num_sources']} relevant chunks):")
            print("="*60)
            for i, source in enumerate(result["sources"], 1):
                print(f"\n[Source {i}] Page {source['page']} | Relevance: {source['relevance_score']}")
                print(source["content"])
