from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import traceback

# Configuration constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "bge-m3:latest"
LLM_MODEL = "deepseek-r1:8b"
FAISS_INDEX_PATH = "vector_store.faiss"
TOP_K_RETRIEVAL = 15
SIMILARITY_THRESHOLD = 0.7
PDF_PATH = "bhagavad_gita.pdf"

# Step 1: Load and chunk documents with improved parameters
print("Loading documents...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,  # Increased from 500 for better context
    chunk_overlap=CHUNK_OVERLAP,  # Increased from 100 for continuity
    separators=["\n\n", "\n", ".", " "],  # Added period separator
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 2: Initialize embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Step 3: Load or create vector store
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully!")
else:
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
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved!")

# Step 4: Initialize LLM
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.3,  # Reduced from 0.6 for more consistent answers
)

# Step 5: Create retriever with EmbeddingsFilter (more efficient than LLMChainExtractor)
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_RETRIEVAL}  # Retrieve more initially
)

# Use EmbeddingsFilter instead of LLMChainExtractor for efficiency
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=SIMILARITY_THRESHOLD
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)

# Step 6: Define strict Gita-only prompt
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

# Step 7: Build LCEL chain (modern approach replacing RetrievalQA)
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

# Step 8: Query function with source tracking
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
                "page": doc.metadata.get("page", "Unknown")
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
        if "error" not in result:
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
                print(f"\n[Source {i}] Page {source['page']}:")
                print(source["content"])
