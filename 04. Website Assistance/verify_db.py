def verify_vector_database():
    # 1. Load the vector database
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.embeddings import OllamaEmbeddings
    
    print("\nLoading vector database...")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vector_store = FAISS.load_local("vector_store.faiss", embeddings=embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector database loaded successfully")
    
    # 2. Verify basic statistics
    print("\nDatabase Statistics:")
    print(f"- Number of documents: {len(vector_store.index_to_docstore_id)}")
    
    # 3. Test retrieval with sample queries
    test_queries = [
        "National Pension System",
        "NPS account opening",
        "Tier I and Tier II accounts",
        "Pension fund managers",
        "Withdrawal rules"
    ]
    
    print("\nRunning test queries:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content[:150]}...")
    
    # 4. Verify content sources
    print("\nChecking content sources:")
    sample_docs = vector_store.similarity_search("NPS", k=1)
    if sample_docs:
        metadata = sample_docs[0].metadata
        print(f"- Document source: {metadata.get('source', 'Unknown')}")
        print(f"- Document length: {len(sample_docs[0].page_content)} characters")
    
    # 5. Check chunking quality
    print("\nChunk quality analysis:")
    long_docs = vector_store.similarity_search("", k=5)  # Get random docs
    for i, doc in enumerate(long_docs):
        content = doc.page_content
        print(f"Chunk {i+1}: {len(content.split())} words, {len(content)} chars")
        if len(content) < 500:
            print("⚠️ Warning: Short chunk detected")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    verify_vector_database()