from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama   
from agno.models.groq import Groq

from dotenv import load_dotenv
import os 
load_dotenv()

# Set the API key for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Initialize the Groq model
model = Groq(id="llama-3.1-8b-instant")


## web search agent
web_search_agent=Agent(model=model,
    name="Web Search Agent",
    role="Search the web for the information",
    tools=[DuckDuckGoTools()],
    instructions=["Alway include sources"],
    show_tool_calls=True,
    markdown=True,

)

## Financial agentß
finance_agent=Agent(model=model,
    name="Finance AI Agent",
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

multi_ai_agent=Agent(model=model,
    name="Multi AI Agent",
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)

