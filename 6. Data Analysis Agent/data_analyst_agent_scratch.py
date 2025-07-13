import streamlit as st
import pandas as pd
import duckdb
import ollama
import json
import tempfile
import os
import re
from datetime import datetime
from langchain_community.llms import Ollama

# DuckDB Agent Implementation
class DuckDbAgentDeepSeek:
    def __init__(self, semantic_model: dict):
        self.semantic_model = self._enhance_semantic_model(semantic_model)
        self.conn = duckdb.connect(database=':memory:')
        self.last_generated_sql = ""
        self.last_response = ""
        self.model = Ollama(model="deepseek-r1:8b", base_url="http://localhost:11434")

    def _enhance_semantic_model(self, model: dict) -> dict:
        for table in model.get("tables", []):
            for col in table.get("columns", []):
                col["duckdb_type"] = self._map_type(col.get("type", "VARCHAR"))
        return model

    def _map_type(self, generic_type: str) -> str:
        type_map = {
            "int": "INTEGER",
            "float": "DOUBLE",
            "str": "VARCHAR",
            "datetime": "TIMESTAMP",
            "bool": "BOOLEAN"
        }
        return type_map.get(generic_type.lower(), "VARCHAR")

    def load_data(self, data: pd.DataFrame, table_name: str):
        self.conn.register(table_name, data)
        return f"Loaded {len(data)} rows into table '{table_name}'"

    def _get_schema_context(self) -> str:
        if not self.semantic_model:
            return "No schema information available"
        
        context = "# Database Schema\n\n"
        
        for table in self.semantic_model.get("tables", []):
            context += f"## Table: `{table['name']}`\n"
            context += f"{table.get('description', 'No description available')}\n\n"
            
            if table.get("columns"):
                context += "### Columns\n"
                for col in table["columns"]:
                    context += f"- `{col['name']}` ({col.get('type', 'unknown')})"
                    if "description" in col:
                        context += f": {col['description']}"
                    context += "\n"
            
            context += "\n"
        
        return context

    def generate_sql(self, query: str) -> str:
        schema_context = self._get_schema_context()
        prompt = f"""
        You are an expert SQL developer. Generate DuckDB SQL queries to solve data questions.
        Return ONLY SQL code in ```sql ``` blocks.
        
        {schema_context}
        
        ### User Query:
        {query}
        
        ### Instructions:
        1. Generate valid DuckDB SQL
        2. Use ONLY the provided schema
        3. Return ONLY the SQL enclosed in ```sql ``` blocks
        4. Use explicit table/column names
        """

        try:
            response = self.model.invoke(prompt)
            print(f"response is {response}")
            response = str(response).strip()
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            print(f"response clean is {response}")
            
            self.last_response = response
            sql_content = response
            
            # Extract SQL from code block if present
            if "```sql" in sql_content:
                sql_content = sql_content.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_content:
                sql_content = sql_content.split("```")[1].split("```")[0].strip()

            # Fallback: remove any markdown or non-SQL text
            sql_content = sql_content.strip()

            self.last_generated_sql = sql_content
            if not sql_content:
                return "Error: No SQL code generated."
            print(f"sql_content is {sql_content}")
            return sql_content
        
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_sql(self, sql: str) -> pd.DataFrame:
        if not sql or "Error:" in sql or not sql.strip():
            return pd.DataFrame({"error": [sql or "No SQL to execute"]})
        
        try:
            return self.conn.execute(sql).fetchdf()
        except Exception as e:
            return pd.DataFrame({"error": [f"SQL execution error: {str(e)}"]})

    def explain_sql(self) -> str:
        if not self.last_generated_sql:
            return "No SQL available to explain"
        
        prompt = f"""
        Explain this DuckDB SQL query in simple terms:
        ```sql
        {self.last_generated_sql}
        ```
        
        Break down:
        1. What tables and columns are used
        2. What operations are performed
        3. What the final result represents
        """
        
        try:
            response = self.model.invoke(prompt)
            response = response[0]["content"].strip()
            return response
        except Exception as e:
            return f"Explanation error: {str(e)}"

    def close(self):
        self.conn.close()

# Streamlit Application
def main():
    st.set_page_config(
        page_title="DeepSeek Data Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 DeepSeek Data Assistant")
    st.caption("Upload your data and query it using natural language powered by DeepSeek")
    
    # Initialize session state
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar for file upload
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    # Read file based on type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:  # Excel
                        df = pd.read_excel(uploaded_file)
                    
                    # Generate semantic model
                    semantic_model = {
                        "tables": [
                            {
                                "name": "uploaded_data",
                                "description": "User uploaded dataset",
                                "columns": [
                                    {
                                        "name": col,
                                        "type": str(df[col].dtype),
                                        "description": f"Column: {col}"
                                    } for col in df.columns
                                ]
                            }
                        ]
                    }
                    
                    # Initialize agent
                    st.session_state.agent = DuckDbAgentDeepSeek(semantic_model)
                    st.session_state.agent.load_data(df, "uploaded_data")
                    st.session_state.data_loaded = True
                    
                    # Show success message
                    st.success(f"Successfully loaded {len(df)} rows!")
                    st.download_button(
                        label="Download Sample Data",
                        data=df.head(100).to_csv(index=False),
                        file_name="sample_data.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.session_state.data_loaded = False
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("Please upload a data file to begin querying")
        st.image("https://i.imgur.com/5XyWYFm.png", caption="DeepSeek Data Assistant")
        return
    
    # Query interface
    st.subheader("Ask about your data")
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., Show total sales by product category",
        height=100
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Run Query", use_container_width=True):
            if not query:
                st.warning("Please enter a question")
                return
            
            if not st.session_state.agent:
                st.error("Please upload a file first to initialize the agent")
                return
            
            with st.spinner("Generating SQL with DeepSeek..."):
                try:
                    # Generate and execute SQL
                    sql = st.session_state.agent.generate_sql(query)
                    results = st.session_state.agent.execute_sql(sql)
                    
                    # Add to history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.insert(0, {
                        "time": timestamp,
                        "query": query,
                        "sql": sql,
                        "results": results
                    })
                    
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
        
        if st.button("Explain SQL", use_container_width=True):
            if not st.session_state.agent:
                st.error("Please upload a file first")
                return
            if st.session_state.agent.last_generated_sql:
                with st.spinner("Generating explanation..."):
                    explanation = st.session_state.agent.explain_sql()
                    st.session_state.history.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "query": "Explain SQL",
                        "explanation": explanation
                    })
    
    with col2:
        if st.session_state.history:
            st.subheader("Query History")
            for item in st.session_state.history:
                with st.expander(f"{item['time']} - {item['query'][:30]}..."):
                    if "sql" in item:
                        st.code(f"SQL:\n{item['sql']}", language="sql")
                        
                        if "error" in item['results'].columns:
                            st.error(item['results']['error'][0])
                        else:
                            st.dataframe(item['results'])
                    elif "explanation" in item:
                        st.markdown("### SQL Explanation")
                        st.write(item['explanation'])

    # Data preview section
    if st.session_state.agent:
        st.subheader("Data Preview")
        preview_size = st.slider("Preview rows:", 5, 50, 10)
        try:
            preview = st.session_state.agent.execute_sql(
                f"SELECT * FROM uploaded_data LIMIT {preview_size}"
            )
            st.dataframe(preview)
        except:
            st.warning("Could not generate preview")
    else:
        st.info("Upload a file to see data preview")

# Run the app
if __name__ == "__main__":
    main()