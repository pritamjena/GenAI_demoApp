



from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama   
from agno.models.groq import Groq
import phi
import os

from agno.playground import Playground, serve_playground_app
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGoTools()],
    instructions=["Alway include sources"],
    show_tool_calls=True,
    markdown=True,

)

## Financial agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

app=Playground(agents=[finance_agent,web_search_agent],
                       app_id="agents-from-scratch-playground-app",
        name="Agents from Scratch Playground",).get_app()

if __name__=="__main__":
    playground= Playground(agents=[finance_agent,web_search_agent],
                       app_id="agents-from-scratch-playground-app",
        name="Agents from Scratch Playground",)
    app=playground.get_app()
    # serve_playground_app("playground:app",reload=True)


if __name__ == "__main__":
    playground.serve(app="playground:app", reload=True)


 # Add endpoint as https://localhost:7777/v1 manually in https://app.agno.com/playground/agents if needed
