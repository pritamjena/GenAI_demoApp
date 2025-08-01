import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import re
from bs4 import Tag
import logging
logging.basicConfig(
    level=logging.INFO,
    filename="web_assist.log",   # Log file name
    filemode="w",                # Overwrite log file each run; use "a" to append
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_url_content(url):
    """Fetch and clean HTML content from a URL, including all tab contents if present."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button"]):
            element.decompose()

        # Try to extract all tab contents if present
        tab_contents = []
        # Common tab content containers
        tab_content_selectors = [
            '.tab-content',
            '[role="tabpanel"]',
            '.ui-tabs-panel',
            '.tabs-panel',
        ]
        found_tab_content = False
        for selector in tab_content_selectors:
            for tab_content in soup.select(selector):
                found_tab_content = True
                # Get all text from each tab pane
                text = tab_content.get_text(separator='\n', strip=True)
                text = re.sub(r'\s+', ' ', text)
                if text:
                    tab_contents.append(text)
        if found_tab_content and tab_contents:
            # If tab contents found, return all joined
            return '\n\n'.join(tab_contents)
        # Fallback: Get all text
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return ""


def extract_links(url: str) -> list[str]:
    """Extract all absolute URLs from a page, including links in tab navigation."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        links = set()
        # Normal anchor tags
        for link in soup.find_all('a', href=True):
            if isinstance(link, Tag):
                href = link.get('href')
                if isinstance(href, str) and not href.startswith(('javascript:', 'mailto:')):
                    absolute_url = urljoin(url, href)
                    if absolute_url.startswith(url):  # Only include same-domain URLs
                        links.add(absolute_url)
        # Also look for tab navigation links (e.g., data-toggle="tab" or role="tab")
        tab_nav_selectors = [
            '[data-toggle="tab"]',
            '[role="tab"]',
            '.nav-tabs a',
            '.tabs-nav a',
        ]
        for selector in tab_nav_selectors:
            for tab_link in soup.select(selector):
                href = tab_link.get('href')
                if isinstance(href, str) and not href.startswith(('javascript:', 'mailto:')):
                    absolute_url = urljoin(url, href)
                    if absolute_url.startswith(url):
                        links.add(absolute_url)
        return list(links)
    except Exception as e:
        logging.error(f"Error extracting links from {url}: {str(e)}")
        return []

def main():
    # Configuration
    start_url = "https://enps.nsdl.com/eNPS/NationalPensionSystem.html"
    output_file = "vector_store.faiss"
    
    logging.info("Starting web scraping...")
    
    # Step 1: Fetch all links from the main URL
    logging.info(f"Fetching links from: {start_url}")
    all_urls = [start_url] + extract_links(start_url)
    logging.info(f"Found {len(all_urls)} URLs to process")
    
    # Step 2: Fetch content from all URLs
    all_content = []
    for i, url in enumerate(all_urls):
        logging.info(f"Processing URL {i+1}/{len(all_urls)}: {url}")
        content = fetch_url_content(url)
        if content:
            all_content.append(content)
    
    # Combine all content
    full_text = "\n\n".join(all_content)
    logging.info(f"Total content size: {len(full_text)} characters")
    
    # Step 3: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    logging.info(f"Created {len(chunks)} text chunks")
    
    # Step 4: Create vector store
    logging.info("Creating embeddings and vector store...")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # Step 5: Save to disk
    vector_store.save_local(output_file)
    logging.info(f"Vector database saved to {output_file}")
    
    # Test retrieval
    query = "What is the National Pension System?"
    docs = vector_store.similarity_search(query, k=3)
    logging.info("\nTest query results:")
    for i, doc in enumerate(docs):
        logging.info(f"\nDocument {i+1}:\n{doc.page_content[:300]}...")

def answer_query(query: str, db_path: str = "vector_store.faiss") -> str:
    """
    Answer a question using the FAISS vector store and Ollama LLM.
    """
    # Load embeddings and vector store
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vector_store = FAISS.load_local(db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    logging.info("✅ Vector database loaded successfully")
    
    # 2. Verify basic statistics
    logging.info("\nDatabase Statistics:")
    logging.info(f"- Number of documents: {len(vector_store.index_to_docstore_id)}")
        
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use Ollama LLM to answer based on context
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3.1:latest", base_url="http://localhost:11434")
    prompt = (
        "Answer the following question based on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = llm.invoke(prompt)
    return answer

if __name__ == "__main__":
    main()
    # Example Q&A
    user_question = "How can I open an NPS account?"
    print("Q:", user_question)
    print("A:", answer_query(user_question))