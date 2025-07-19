from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add Ollama LLM and RetrievalQA chain for tax assistant
# from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="deepseek-r1:8b")

urls=[
    "https://tax2win.in/guide/how-to-calculate-income-tax-on-salary",
    "https://cleartax.in/paytax/taxcalculator",
]

docs=[WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

## Add alll these text to vectordb
embeddings=(
    OllamaEmbeddings(model="deepseek-r1:8b")  ##by default it ues llama2
)

vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
print("Vector DB done")


retriever=vectorstore.as_retriever()

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
    sources = result.get("source_documents", [])
    return answer, sources
    
answer, sources = ask_tax_assistant("How do I calculate tax on my salary?")
print("Answer:", answer)