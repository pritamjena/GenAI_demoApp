"""
Ollama FastAPI Server
---------------------
This FastAPI application provides an API interface to interact with Ollama models for chat and text generation. It includes endpoints to list available models, perform chat completions, generate text completions, and check server health.

Modules:
- FastAPI: Web framework for building APIs
- Pydantic: Data validation and settings management
- httpx: Async HTTP client for making requests to Ollama server
- os, dotenv: For environment variable management

Endpoints:
- /models: List available Ollama models
- /chat: Chat completion with a model
- /generate: Text generation with a model
- /health: Health check for the API server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from typing import Optional


load_dotenv()

app = FastAPI(
    title="Ollama FastAPI Server",
    description="API for interacting with Ollama models",
    version="0.1.0"
)


# Configuration
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:latest")


class ChatRequest(BaseModel):
    """
    Request body for chat completion endpoint.
    
    Attributes:
        prompt (str): The user's input prompt for the chat model.
        model (Optional[str]): The model to use for chat (default from environment).
        stream (Optional[bool]): Whether to stream the response (default False).
        context (Optional[list]): Optional context for the chat session.
    """
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    stream: Optional[bool] = False
    context: Optional[list] = None


class GenerateRequest(BaseModel):
    """
    Request body for text generation endpoint.
    
    Attributes:
        prompt (str): The prompt for text generation.
        model (Optional[str]): The model to use for generation (default from environment).
        stream (Optional[bool]): Whether to stream the response (default False).
    """
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    stream: Optional[bool] = False

@app.get("/models")
async def list_models():
    """
    List available Ollama models.
    
    Returns:
        dict: A dictionary containing available model tags from the Ollama server.
    Raises:
        HTTPException: If the Ollama server is not reachable.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=500, detail="Ollama server not reachable")


@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint.
    
    Args:
        request (ChatRequest): The chat request payload.
    Returns:
        dict: The chat completion response from the Ollama server.
    Raises:
        HTTPException: If the Ollama server is not reachable.
    """
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": request.stream
    }
    if request.context:
        payload["context"] = request.context

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=180.0
            )
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=500, detail="Ollama server not reachable")

@app.post("/generate")
async def generate_completion(request: GenerateRequest):
    """
    Generate completion endpoint.
    
    Args:
        request (GenerateRequest): The text generation request payload.
    Returns:
        dict: The text generation response from the Ollama server.
    Raises:
        HTTPException: If the Ollama server is not reachable.
    """
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": request.stream
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=180.0
            )
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=500, detail="Ollama server not reachable")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Status of the API server.
    Raises:
        HTTPException: If the Ollama server is not reachable.
    """
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{OLLAMA_URL}")
            return {"status": "healthy"}
        except httpx.ConnectError:
            raise HTTPException(status_code=500, detail="Ollama server not reachable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)