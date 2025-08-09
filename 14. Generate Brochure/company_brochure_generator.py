#!/usr/bin/env python3
"""
Company Brochure Generator
A script that scrapes company websites and generates brochures using Ollama LLM.
"""

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
            print(f"Failed to fetch {self.url}: {str(e)}")
            self.body = b""
            self.title = "Error loading page"
            self.text = f"Failed to load website: {str(e)}"
            self.links = []
        except Exception as e:
            print(f"Unexpected error loading {self.url}: {str(e)}")
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
        print(f"URL validation error: {str(e)}")
        return False

def extract_json_from_text(text):
    """
    Extract the first JSON object found in the text.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

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
    if is_url_reachable(url, 5):
        result = "Landing page:\n"
        result += Website(url).get_contents()
        links = get_links(url)
        print("Found links:", links)
        if links and "links" in links:
            for link in links["links"]:
                result += f"\n\n{link['type']}\n"
                result += Website(link["url"]).get_contents()
        return result
    return None

def get_brochure_user_prompt(company_name, url):
    """Generate user prompt for brochure creation"""
    try:
        if is_url_reachable(url):
            web_content = get_all_details(url)
            if web_content:
                web_content = web_content[:5000]  # Truncate if more than 5,000 characters
                user_prompt = f"You are looking at a company called: {company_name}\n"
                user_prompt += f"Use the name {company_name} clearly in the brochure.\n"
                user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
                user_prompt += f"\n\nReminder: the company name is {company_name}."
                user_prompt += web_content
                return user_prompt
    except requests.RequestException:
        return False
    return None

def create_brochure(company_name, url):
    """Create a brochure for a company"""
    system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

    # Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':
    # system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
    # and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
    # Include details of company culture, customers and careers/jobs if you have the information."

    try:
        if is_url_reachable(url, 5):
            user_prompt = get_brochure_user_prompt(company_name, url)
            if user_prompt:
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
        
                result = response.choices[0].message.content
                return result
        else:
            print(f"❌ URL {url} is not reachable")
            return None
    except requests.RequestException as e:
        print(f"❌ Error accessing URL: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def stream_brochure(company_name, url):
    """Stream brochure generation with real-time output"""
    if not is_url_reachable(url):
        print("❌ URL not reachable")
        return
    
    system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

    try:
        user_prompt = get_brochure_user_prompt(company_name, url)
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
            print("Generating brochure...\n")
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response += content.replace("```", "")
                    print(content, end='', flush=True)
            
            return response
    except requests.RequestException as e:
        print(f"❌ Error accessing URL: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    """Main function to demonstrate the brochure generator"""
    print("Company Brochure Generator")
    print("=" * 50)
    
    # Example usage
    company_name = "HuggingFace"
    url = "https://huggingface.co"
    
    print(f"Generating brochure for {company_name}...")
    print(f"URL: {url}")
    print("\n" + "=" * 50)
    
    # Option 1: Create brochure without streaming
    print("Option 1: Generate brochure (non-streaming)")
    brochure = create_brochure(company_name, url)
    if brochure:
        print(brochure)
    
    print("\n" + "=" * 50)
    
    # Option 2: Stream brochure generation
    print("Option 2: Generate brochure (streaming)")
    stream_brochure(company_name, url)

if __name__ == "__main__":
    main()
