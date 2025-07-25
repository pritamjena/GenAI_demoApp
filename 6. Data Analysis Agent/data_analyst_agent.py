import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from agno.models.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools
import re
from phi.model.groq.groq import Groq
from phi.model.ollama.chat import Ollama


from dotenv import load_dotenv
import os 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is not None:
    os.environ["GROQ_API_KEY"] = groq_api_key

model = Groq(id="llama-3.1-8b-instant")

model_ollama = Ollama(id="deepseek-r1:8b")

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    # groq_key = st.text_input("Enter your Groq API key:", type="password")
    groq_key = groq_api_key
    if groq_key:
        st.session_state.groq_key = groq_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your Groq API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "groq_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        
        # Initialize the DuckDbAgent for SQL query generation
        duckdb_agent = DuckDbAgent(
            provider=model,  # Use model object, not string
            agent_id="data-analyst-agent",
            session_id="default-session",
            knowledge_base=None,
            add_references=False,
            references_format="json",
            output_model=None,
            debug_mode=False,
            semantic_model=json.dumps(semantic_model),
            # tools=None,  # Remove PandasTools() since it's not compatible
            # tools=[PandasTools()],
            tool_choice="auto",
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
        )
        
        # Initialize code storage in sessionß state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from DuckDbAgent
               
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                        user_query,
                        stream=True,
                        )

                    # Display the response in Streamlit
                    st.markdown(response_content)
                
                    
                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")