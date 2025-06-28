import requests
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os 
import re
load_dotenv()
import warnings
import json
warnings.filterwarnings("ignore")


RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# Step 1: Call the RapidAPI weather endpoint
def amazon_deal():
    url = "https://real-time-amazon-data.p.rapidapi.com/deals-v2"

    querystring = {"country":"IN","min_product_star_rating":"4","price_range":"ALL","discount_range":"ALL"}

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)  
    data = response.json()

    print(f"API Data is {data}")
    return data 

def summarize_deal(deal_data):
    # Extract fields from the JSON

    final_deal = []
    deals = deal_data.get('data', {}).get('deals', [])
    for item in deals:
        deal= {
            "title" : item.get('deal_title', 'No Title'),
            "deal_price" : item.get('deal_price', {}).get('amount', 'No amount'),
            "savings" : item.get('savings_amount', {}).get('amount', 'No amount'),
            "deal_end" : item.get('deal_ends_at', 'No end time')
                }
        final_deal.append(deal)

    print(f"Final Deal is {final_deal}")

    prompt = f"""
        You are a helpful amazon best deals provider assistant. Summarize the current deal data for a general audience, .

        Here is the deal data {final_deal} that has title,deal_price,savings,deal_end_str

        Summarise the deals in pointers and deal information shuould be crisp clear without providing lots of information and sorts the suggestion based on savings.
        """
    # Use Ollama DeepSeek model to summarize deal info
    model = Ollama(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434",  
        
    )

    result = model(prompt)
    clean_response = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return clean_response.strip()

# Example usage
if __name__ == "__main__":

    deal_data = amazon_deal()
    print("######## DEALS AHEAD ########")
    summary = summarize_deal(deal_data)
    print(summary)