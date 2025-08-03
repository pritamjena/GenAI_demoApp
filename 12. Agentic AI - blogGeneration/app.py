import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

import os
from dotenv import load_dotenv
load_dotenv()

app=FastAPI()

# print(os.getenv("LANGCHAIN_API_KEY"))

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Simple request model
class BlogRequest(BaseModel):
    topic: str
    generate_language: str = "English"  # Default to English if not specified

## API's

@app.post("/blogs")
async def create_blogs(request: BlogRequest):
    
    topic = request.topic
    generate_language = request.generate_language

    ## get the llm object

    groqllm=GroqLLM()
    llm=groqllm.get_llm()

    ## get the graph
    graph_builder=GraphBuilder(llm)
    if topic:
        graph=graph_builder.setup_graph(usecase="topic")
        state=graph.invoke({"topic": topic, "generate_language": generate_language})

    return {"data":state}

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)

