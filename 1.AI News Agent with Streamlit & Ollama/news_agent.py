import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType

import os

# Load env variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit UI
st.title("🌐 AI News Agent")
st.caption("Search the latest news with your local LLM")

# Initialize LLM (adjust model name as needed)
llm = Ollama(
    model="deepseek-r1:8b", 
    base_url="http://localhost:11434"
)

# Tool: DuckDuckGo search
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Useful for searching web for current events and news."
    )
]
# tool_names = ", ".join([tool.name for tool in tools])
tool_names = ", ".join([tool.name for tool in tools])
tools_string = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])  # 🛠️ THIS IS THE FIX

print(f" tool name is {tools}")
print(f" tool names are {tool_names}")
print(f" tools string is {tools_string}")

# Prompt Template for summarization
CUSTOM_PROMPT="""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


# Declare all required variables in PromptTemplate
prompt = PromptTemplate(
    template=CUSTOM_PROMPT,
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"]
)

# Create agent and executor
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    # max_iterations=3,
    handle_parsing_errors=True,
    verbose=True,
    # early_stopping_method="generate"
)


# UI Input
query = st.text_input("What news would you like to research?", "Latest AI developments")

if st.button("Search"):
    with st.spinner("Thinking..."):
            # Manually supply all template variables required by prompt
        agent_input = {
                "input": query,
                "tools": tools_string,
                "tool_names": tool_names,
                "agent_scratchpad": ""
            }
        result = agent_executor.invoke(agent_input)

        st.subheader("🔍 Final Answer")
        st.write(result["output"])

            # with st.expander("🧾 Raw Search Data"):
            #     raw_data = search.run(query)
            #     st.write(raw_data)
            


