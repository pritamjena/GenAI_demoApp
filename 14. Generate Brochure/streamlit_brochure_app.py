#!/usr/bin/env python3
"""
Streamlit Company Brochure Generator
A web interface for generating company brochures using Ollama LLM.
"""

import streamlit as st
import os
import requests
import json
import logging
import re
import pprint
from typing import List
from bs4 import BeautifulSoup
from openai import OpenAI

# Initialize OpenAI client for Ollama
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
MODEL = "deepseek-r1:8b"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Headers for web scraping
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = normalize_url(url)
        try:
            response = requests.get(self.url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            self.body = response.content
            soup = BeautifulSoup(self.body, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"
            if soup.body:
                for irrelevant in soup.body(["script", "style", "img", "input"]):
                    irrelevant.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            else:
                self.text = ""
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute URLs
                    if href.startswith('/'):
                        href = f"{self.url.rstrip('/')}{href}"
                    elif not href.startswith(('http://', 'https://')):
                        href = f"{self.url.rstrip('/')}/{href.lstrip('/')}"
                    links.append(href)
            self.links = links
        except requests.RequestException as e:
            st.error(f"Failed to fetch {self.url}: {str(e)}")
            self.body = b""
            self.title = "Error loading page"
            self.text = f"Failed to load website: {str(e)}"
            self.links = []
        except Exception as e:
            st.error(f"Unexpected error loading {self.url}: {str(e)}")
            self.body = b""
            self.title = "Error loading page"
            self.text = f"Unexpected error: {str(e)}"
            self.links = []

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def normalize_url(url):
    """Normalize URL by adding https:// if no protocol is specified"""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def is_url_reachable(url, timeout=10):
    """Check if a URL is reachable with better error handling"""
    try:
        # Normalize the URL first
        url = normalize_url(url)
        
        # Try HEAD request first (faster)
        try:
            response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
            if response.status_code < 400:
                return True
        except requests.RequestException:
            pass
        
        # If HEAD fails, try GET request
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            return response.status_code < 400
        except requests.RequestException:
            return False
            
    except Exception as e:
        st.warning(f"URL validation error: {str(e)}")
        return False

def extract_json_from_text(text):
    """
    Extract the first JSON object found in the text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def remove_think_tags(text):
    """
    Remove <think></think> tags and their content using regex
    """
    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also remove any remaining <think> or </think> tags
    text = re.sub(r'</?think>', '', text)
    return text.strip()

def get_links_user_prompt(website):
    """Generate user prompt for link extraction"""
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Include links to About, Services, Industries, Careers, Insights, Blog, Case Studies, and Company pages. \
Do not include Terms of Service, Privacy, email links, or social media links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    """Extract relevant links from a website"""
    website = Website(url)
    
    link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company. \
Look for links related to: About Us, Company, Services, Industries, Careers, Jobs, Insights, Blog, Case Studies, \
Our Story, What We Do, Solutions, Products, Team, Leadership, News, Press, and similar business-relevant pages.\n"
    link_system_prompt += "You should respond in JSON as in this example:"
    link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "services page", "url": "https://full.url/services"},
        {"type": "careers page", "url": "https://another.full.url/careers"},
        {"type": "industries page", "url": "https://full.url/industries"}
    ]
}
"""
    
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": link_system_prompt},
                {"role": "user", "content": get_links_user_prompt(website)}
            ]
        )

        result = response.choices[0].message.content
       
        # Log the raw result for debugging
        logging.debug(f"Raw result: {result}")

       
        if isinstance(result, str):
            if not result.strip():
                logging.warning("Result string is empty.")
                return None

            json_text = extract_json_from_text(result)
            if not json_text:
                logging.warning("No JSON object found in the result string.")
                return None

            logging.debug(f"Extracted JSON string: {repr(json_text)}")

            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
                logging.debug(f"Problematic JSON string: {repr(json_text)}")
                return None
        
    except Exception as e:
        logging.exception("An unexpected error occurred in get_links.")
        return None

def get_all_details(url):
    """Get all relevant details from a website including landing page and linked pages"""
    try:
        # Normalize URL first
        normalized_url = normalize_url(url)
        
        # Check if URL is reachable
        if not is_url_reachable(normalized_url, 10):
            st.error(f"❌ URL {normalized_url} is not reachable")
            return None
            
        result = "Landing page:\n"
        website = Website(normalized_url)
        result += website.get_contents()
        
        # Only try to get links if the website loaded successfully
        if website.text and not website.text.startswith("Failed to load"):
            # Debug: Show all found links
            st.info(f"Found {len(website.links)} total links on the website")
            if len(website.links) > 0:
                st.write("Sample links found:", website.links[:10])  # Show first 10 links
            
            links = get_links(normalized_url)
            if links and "links" in links:
                st.success(f"✅ Found {len(links['links'])} relevant links")
                for link in links["links"]:
                    try:
                        result += f"\n\n{link['type']}\n"
                        link_website = Website(link["url"])
                        result += link_website.get_contents()
                    except Exception as e:
                        st.warning(f"Failed to load link {link['url']}: {str(e)}")
                        continue
            else:
                st.warning("❌ No relevant links found by AI")
                st.info("This might be due to the AI not recognizing relevant links or the website structure")
                
                # Fallback: Try to manually identify some common link patterns
                fallback_links = []
                for link in website.links:
                    link_lower = link.lower()
                    if any(keyword in link_lower for keyword in ['about', 'company', 'services', 'industries', 'careers', 'insights', 'blog', 'case-studies']):
                        link_type = "relevant page"
                        if 'about' in link_lower:
                            link_type = "about page"
                        elif 'services' in link_lower:
                            link_type = "services page"
                        elif 'industries' in link_lower:
                            link_type = "industries page"
                        elif 'careers' in link_lower:
                            link_type = "careers page"
                        elif 'insights' in link_lower or 'blog' in link_lower:
                            link_type = "insights page"
                        
                        fallback_links.append({"type": link_type, "url": link})
                
                if fallback_links:
                    st.info(f"🔧 Using fallback method: Found {len(fallback_links)} relevant links")
                    for link in fallback_links:
                        try:
                            result += f"\n\n{link['type']}\n"
                            link_website = Website(link["url"])
                            result += link_website.get_contents()
                        except Exception as e:
                            st.warning(f"Failed to load fallback link {link['url']}: {str(e)}")
                            continue
        else:
            st.warning("Website content could not be loaded properly")
            
        return result
    except Exception as e:
        st.error(f"Error processing URL {url}: {str(e)}")
        return None

def get_brochure_user_prompt(company_name, url):
    """Generate user prompt for brochure creation"""
    try:
        # Normalize URL first
        normalized_url = normalize_url(url)
        
        if is_url_reachable(normalized_url):
            web_content = get_all_details(normalized_url)
            if web_content:
                web_content = web_content[:5000]  # Truncate if more than 5,000 characters
                user_prompt = f"You are looking at a company called: {company_name}\n"
                user_prompt += f"Use the name {company_name} clearly in the brochure.\n"
                user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
                user_prompt += f"\n\nReminder: the company name is {company_name}."
                user_prompt += web_content
                return user_prompt
            else:
                st.error("Failed to get website content")
                return None
        else:
            st.error(f"URL {normalized_url} is not reachable")
            return None
    except Exception as e:
        st.error(f"Error creating user prompt: {str(e)}")
        return None

def create_brochure(company_name, url):
    """Create a brochure for a company"""
    system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

    try:
        # Normalize URL first
        normalized_url = normalize_url(url)
        
        if is_url_reachable(normalized_url, 10):
            user_prompt = get_brochure_user_prompt(company_name, normalized_url)
            if user_prompt:
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
        
                result = response.choices[0].message.content
                # Remove think tags from the result
                result = remove_think_tags(result)
                return result
            else:
                st.error("Failed to create user prompt")
                return None
        else:
            st.error(f"❌ URL {normalized_url} is not reachable")
            return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

def stream_brochure(company_name, url):
    """Stream brochure generation with real-time output"""
    try:
        # Normalize URL first
        normalized_url = normalize_url(url)
        
        if not is_url_reachable(normalized_url):
            st.error(f"❌ URL {normalized_url} not reachable")
            return None
        
        system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

        user_prompt = get_brochure_user_prompt(company_name, normalized_url)
        if user_prompt:
            stream = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
        
            response = ""
            st.info("Generating brochure...")
            
            # Create a placeholder for streaming content in the main area
            with st.container():
                st.markdown("### 📋 Generated Content")
                placeholder = st.empty()
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        response += content
                        # Remove think tags from the accumulated response
                        clean_response = remove_think_tags(response)
                        # Update the placeholder with the clean content
                        placeholder.markdown(clean_response)
            
            return clean_response
        else:
            st.error("Failed to create user prompt for streaming")
            return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Company Brochure Generator",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Company Brochure Generator")
    st.markdown("Generate professional brochures from company websites using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = ["deepseek-r1:8b", "gemma3:4b"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0
        )
        
        # Update the model
        global MODEL
        MODEL = selected_model
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter the company name
        2. Enter the company website URL
        3. Choose streaming or non-streaming mode
        4. Click Generate Brochure
        """)
        
        st.markdown("### 🔧 Requirements")
        st.markdown("""
        - Ollama must be running
        - Selected model must be pulled
        - Internet connection required
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Company name input
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., HuggingFace, OpenAI, Microsoft",
            help="Enter the name of the company for the brochure"
        )
        
        # URL input
        url = st.text_input(
            "Company Website URL",
            placeholder="https://example.com",
            help="Enter the full URL of the company's website"
        )
        
        # Streaming option
        use_streaming = st.checkbox(
            "Use Streaming Mode",
            value=False,
            help="Enable real-time streaming of brochure generation"
        )
        
        # Generate button
        if st.button("🚀 Generate Brochure", type="primary"):
            if not company_name or not url:
                st.error("Please enter both company name and URL")
            else:
                with st.spinner("Processing..."):
                    if use_streaming:
                        brochure_content = stream_brochure(company_name, url)
                        if brochure_content:
                            st.session_state.brochure_content = brochure_content
                            st.success("✅ Brochure generated successfully!")
                    else:
                        brochure = create_brochure(company_name, url)
                        if brochure:
                            st.session_state.brochure_content = brochure
                            st.success("✅ Brochure generated successfully!")
    
    with col2:
        st.header("📊 Status")
        
        # Check Ollama connection
        try:
            # Simple test to check if Ollama is running
            test_response = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            st.success(f"✅ Ollama connected (Model: {MODEL})")
        except Exception as e:
            st.error(f"❌ Ollama connection failed: {str(e)}")
            st.info("Make sure Ollama is running and the model is pulled")
        
        # URL reachability check
        if url:
            try:
                normalized_url = normalize_url(url)
                if is_url_reachable(normalized_url):
                    st.success(f"✅ URL is reachable: {normalized_url}")
                else:
                    st.error(f"❌ URL is not reachable: {normalized_url}")
            except Exception as e:
                st.error(f"❌ URL validation error: {str(e)}")
        
        # Model info
        st.info(f"**Current Model:** {MODEL}")
        st.info(f"**Streaming Mode:** {'Enabled' if use_streaming else 'Disabled'}")
    
    # Brochure display area
    st.markdown("---")
    st.header("📄 Generated Brochure")
    
    # Create a container for the brochure content
    brochure_container = st.container()
    
    # Store brochure in session state
    if 'brochure_content' not in st.session_state:
        st.session_state.brochure_content = None
    
    # Display brochure if available
    if st.session_state.brochure_content:
        with brochure_container:
            st.markdown("### 📋 Generated Content")
            st.markdown(st.session_state.brochure_content)
            
            # Add download button
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="📥 Download Brochure (Markdown)",
                    data=st.session_state.brochure_content,
                    file_name=f"{company_name}_brochure.md",
                    mime="text/markdown"
                )
            with col2:
                if st.button("🗑️ Clear Brochure"):
                    st.session_state.brochure_content = None
                    st.rerun()
    else:
        with brochure_container:
            st.info("👆 Enter company details and click 'Generate Brochure' to create a brochure")
    
    # Add a separator
    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit and Ollama | Company Brochure Generator</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
