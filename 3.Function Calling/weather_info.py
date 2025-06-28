import requests
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os 
import re
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
import markdown
from bs4 import BeautifulSoup

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# Step 1: Call the RapidAPI weather endpoint
def get_weather(city):
    url = "https://open-weather13.p.rapidapi.com/city"

    querystring = {"city":city,"lang":"EN"}

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "open-weather13.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    return data

# Step 2: Use Ollama DeepSeek model to summarize weather info
def summarize_weather(weather_data):
    # Extract fields from the JSON
    city = weather_data.get('name', 'Unknown')
    country = weather_data.get('sys', {}).get('country', 'Unknown')
    temp = weather_data.get('main', {}).get('temp')
    feels_like = weather_data.get('main', {}).get('feels_like')
    humidity = weather_data.get('main', {}).get('humidity')
    condition = weather_data.get('weather', [{}])[0].get('description', 'N/A')
    wind_speed = weather_data.get('wind', {}).get('speed')
    visibility = weather_data.get('visibility')

    # Build prompt
    prompt = f"""
        You are a helpful assistant. Summarize the current weather report for a general audience, .

        City: {city}
        Country: {country}
        Temperature (°F): {temp}
        Feels Like (°F): {feels_like}
        Condition: {condition}
        Humidity: {humidity}%
        Wind Speed: {wind_speed} mph
        Visibility: {visibility} meters

        Write a short and friendly weather summary and change all °F to °C in final answer.
        """

    model = Ollama(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434",  
        
    )

    result = model(prompt)
    clean_response = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return clean_response.strip()

# Example usage
if __name__ == "__main__":
    city = "cuttack"
    weather = get_weather(city)
    print(f"Weather data for {city}: {weather}")
    summary = summarize_weather(weather)
    print(BeautifulSoup(summary, "html.parser").get_text())

