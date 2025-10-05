# Docker Deployment Guide

This guide explains how to deploy the Bhagavad Gita Chatbot using Docker with separate containers for the API and Ollama services.

## 🏗️ Architecture

The Docker setup uses a multi-container architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Container │    │  Ollama LLM     │    │ Ollama Embed    │
│   (FastAPI)     │◄──►│  (Port 11435)   │    │ (Port 11436)    │
│   Port 8000     │    │  deepseek-r1    │    │ bge-m3:latest   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Internet connection for model downloads
- PDF file: `bhagavad-gita-in-english-source-file.pdf`

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd bhagavad-gita-chatbot
```

### 2. Add PDF File

Place your Bhagavad Gita PDF in the project root:
```bash
# Ensure the PDF file is present
ls -la bhagavad-gita-in-english-source-file.pdf
```

### 3. Start All Services

```bash
# Build and start all containers
docker-compose up --build -d

# Check status
docker-compose ps
```

### 4. Wait for Models to Download

The first startup will download ~2.3GB of models:
- `deepseek-r1:1.5b` (1.1GB) - Language model
- `bge-m3:latest` (1.2GB) - Embedding model

Monitor progress:
```bash
# Watch download progress
docker-compose logs ollama-llm -f
docker-compose logs ollama-embed -f
```

### 5. Verify Everything is Working

```bash
# Check API health
curl http://localhost:8000/health

# Check available models
curl http://localhost:11435/api/tags  # LLM models
curl http://localhost:11436/api/tags  # Embedding models
```

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
services:
  ollama-llm:
    environment:
      - MODEL=deepseek-r1:1.5b  # Change LLM model
    ports:
      - "11435:11434"           # Change port if needed

  ollama-embed:
    environment:
      - MODEL=bge-m3:latest     # Change embedding model
    ports:
      - "11436:11434"           # Change port if needed

  api:
    environment:
      - OLLAMA_LLM_BASE_URL=http://ollama-llm:11434
      - OLLAMA_EMBED_BASE_URL=http://ollama-embed:11434
      - FAISS_INDEX_PATH=/data/vector_store.faiss
```

### Model Configuration

To use different models, update the environment variables:

```yaml
# For smaller/faster models
- MODEL=llama3.2:3b        # LLM (smaller)
- MODEL=nomic-embed-text   # Embedding (smaller)

# For larger/more capable models  
- MODEL=llama3.1:70b       # LLM (larger)
- MODEL=bge-large-en-v1.5  # Embedding (larger)
```

## 📊 Service Status

### Check Container Status

```bash
# View all containers
docker-compose ps

# Expected output:
# NAME               STATUS
# gita-chatbot-api   Up
# ollama-embed       Up  
# ollama-llm         Up
```

### Monitor Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs ollama-llm -f
docker-compose logs ollama-embed -f
docker-compose logs api -f
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Ollama services
curl http://localhost:11435/api/tags
curl http://localhost:11436/api/tags
```

## 🧪 Testing the API

### 1. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. Test with curl

```bash
# Get available Q&A types
curl http://localhost:8000/qa-types

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the meaning of dharma according to Krishna?",
    "qa_type": "Philosophical Inquiry"
  }'
```

### 3. Test with Python

```python
import requests

# Test the API
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What does Krishna teach about karma yoga?",
        "qa_type": "Conceptual Understanding"
    }
)

result = response.json()
print(f"Answer: {result['cleaned_response']}")
print(f"Sources: {result['num_sources']}")
```

## 🔄 Management Commands

### Start Services

```bash
# Start in background
docker-compose up -d

# Start with rebuild
docker-compose up --build -d

# Start specific service
docker-compose up ollama-llm -d
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clears models)
docker-compose down -v
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
```

### Update Services

```bash
# Pull latest images and rebuild
docker-compose pull
docker-compose up --build -d
```

## 🗂️ Data Persistence

### Volumes

The setup uses Docker volumes for data persistence:

```yaml
volumes:
  ollama_llm_models:    # LLM models storage
  ollama_embed_models:  # Embedding models storage  
  api_data:            # FAISS index and app data
```

### Backup Data

```bash
# Backup FAISS index
docker cp gita-chatbot-api:/data/vector_store.faiss ./backup/

# Backup models
docker run --rm -v ollama_llm_models:/data -v $(pwd):/backup alpine tar czf /backup/llm_models.tar.gz -C /data .
```

### Restore Data

```bash
# Restore FAISS index
docker cp ./backup/vector_store.faiss gita-chatbot-api:/data/

# Restore models
docker run --rm -v ollama_llm_models:/data -v $(pwd):/backup alpine tar xzf /backup/llm_models.tar.gz -C /data
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Downloading

**Symptoms**: 404 errors, "model not found"

**Solution**:
```bash
# Check download progress
docker-compose logs ollama-llm -f
docker-compose logs ollama-embed -f

# Wait for completion (5-15 minutes)
# Models are large: 1.1GB + 1.2GB = 2.3GB total
```

#### 2. API Connection Errors

**Symptoms**: `ConnectionError: Failed to connect to Ollama`

**Solution**:
```bash
# Check if services are running
docker-compose ps

# Check service logs
docker-compose logs api

# Restart API after models are ready
docker-compose restart api
```

#### 3. Port Conflicts

**Symptoms**: "Port already in use"

**Solution**:
```bash
# Check what's using the ports
lsof -i :8000
lsof -i :11435
lsof -i :11436

# Change ports in docker-compose.yml
```

#### 4. Out of Memory

**Symptoms**: Containers restarting, slow performance

**Solution**:
```bash
# Check memory usage
docker stats

# Use smaller models
# Edit docker-compose.yml:
# - MODEL=llama3.2:3b
# - MODEL=nomic-embed-text
```

#### 5. FAISS Index Issues

**Symptoms**: "FAISS index corrupted" or slow startup

**Solution**:
```bash
# Remove and rebuild index
docker-compose down
docker volume rm 15gita-basedbot_api_data
docker-compose up -d
```

### Debug Commands

```bash
# Check container resource usage
docker stats

# Check container logs
docker-compose logs [service-name]

# Execute commands in container
docker-compose exec api bash
docker-compose exec ollama-llm ollama list

# Check network connectivity
docker-compose exec api curl http://ollama-llm:11434/api/tags
docker-compose exec api curl http://ollama-embed:11434/api/tags
```

## 🚀 Production Deployment

### Security Considerations

1. **Change default ports**:
```yaml
ports:
  - "8001:8000"  # API
  - "11437:11434" # LLM
  - "11438:11434" # Embed
```

2. **Use reverse proxy** (nginx/traefik):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Enable HTTPS**:
```bash
# Use Let's Encrypt with nginx
certbot --nginx -d your-domain.com
```

### Performance Optimization

1. **Use GPU acceleration** (if available):
```yaml
services:
  ollama-llm:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

2. **Increase memory limits**:
```yaml
services:
  ollama-llm:
    deploy:
      resources:
        limits:
          memory: 4G
```

3. **Use production ASGI server**:
```dockerfile
# In Dockerfile.api, replace uvicorn with gunicorn
CMD ["gunicorn", "fastapi_gita_chatbot:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Monitoring

```bash
# Health check script
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:11435/api/tags || exit 1
curl -f http://localhost:11436/api/tags || exit 1
echo "All services healthy"
```

## 📈 Scaling

### Horizontal Scaling

For high traffic, run multiple API instances:

```yaml
services:
  api:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### Load Balancing

Use nginx or traefik to distribute requests:

```nginx
upstream gita_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location / {
        proxy_pass http://gita_api;
    }
}
```

## 🔧 Development

### Local Development with Docker

```bash
# Mount source code for live reload
docker-compose -f docker-compose.dev.yml up

# Or use bind mounts
docker-compose up -d ollama-llm ollama-embed
python app/fastapi_gita_chatbot.py  # Run API locally
```

### Debug Mode

```bash
# Enable debug logging
docker-compose -f docker-compose.debug.yml up
```

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `0.0.0.0:11434` | Ollama server address |
| `MODEL` | `deepseek-r1:1.5b` | Model to use |
| `OLLAMA_LLM_BASE_URL` | `http://ollama-llm:11434` | LLM service URL |
| `OLLAMA_EMBED_BASE_URL` | `http://ollama-embed:11434` | Embedding service URL |
| `FAISS_INDEX_PATH` | `/data/vector_store.faiss` | FAISS index location |

## 🆘 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs -f`
2. **Verify services**: `docker-compose ps`
3. **Test connectivity**: `curl http://localhost:8000/health`
4. **Check resources**: `docker stats`
5. **Restart services**: `docker-compose restart`

For persistent issues, try:
```bash
# Clean restart
docker-compose down -v
docker-compose up --build -d
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ollama Documentation](https://ollama.ai/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FAISS Documentation](https://faiss.ai/)

---

**Happy Chatting with the Gita! 🙏**
