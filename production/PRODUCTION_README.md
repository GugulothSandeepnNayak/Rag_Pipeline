# RAG Pipeline - Production Deployment Guide

## Overview

This is a **production-grade Retrieval-Augmented Generation (RAG) application** with:
- ✅ FastAPI backend with rate limiting & authentication
- ✅ Structured logging and error handling
- ✅ GPU acceleration (NVIDIA CUDA + Apple Metal + CPU fallback)
- ✅ Response caching for embeddings
- ✅ Configuration management (YAML)
- ✅ Health checks & monitoring endpoints
- ✅ CORS support for distributed systems

## Architecture

```
┌─────────────────────┐
│  FastAPI Backend    │ (main.py)
│  - REST API         │
│  - Rate Limiting    │
│  - Authentication   │
└──────────┬──────────┘
           │
┌──────────v──────────────────┐
│  RAG Pipeline               │ (rag_pipeline_prod.py)
│  - Logging & Error Handle   │
│  - Embedding Cache          │
│  - Hybrid Retrieval         │
│  - LLM Generation           │
└──────────┬──────────────────┘
           │
┌──────────v──────────────────┐
│  External Services          │
│  - Chroma Vector DB         │
│  - Ollama LLM (llama3)       │
│  - Sentence Transformers    │
└─────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
# Production setup
pip install -r requirements_prod.txt

# Install Ollama (if not already installed)
# Download from: https://ollama.ai
```

### 2. Pull LLM Model

```bash
# Download llama3 (4.7 GB)
ollama pull llama3

# Or use a smaller model
ollama pull orca-mini  # ~4 GB
ollama pull neural-chat  # ~4 GB
```

### 3. Configure Application

Edit `config.yaml`:

```yaml
models:
  embedding:
    name: "all-mpnet-base-v2"
    device: "auto"  # or "cuda", "mps", "cpu"
  llm:
    name: "llama3"
    endpoint: "http://localhost:11434"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

security:
  api_key_enabled: false  # set to true for production
```

## Running the Application

### Option 1: Development (Single Worker)

```bash
python main.py
```

### Option 2: Production (Multi-Worker with Uvicorn)

```bash
# 4 worker processes
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With reload (dev only)
uvicorn main:app --reload
```

### Option 3: Docker (Recommended)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t rag-pipeline .
docker run -p 8000:8000 \
  -e RAG_API_KEY="your-secret-key" \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/logs:/app/logs \
  rag-pipeline
```

### Option 4: Run with Ollama in Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_GPU=1  # For GPU support

  rag-api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - ollama
    environment:
      - RAG_API_KEY=your-secret-key
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

volumes:
  ollama_data:
```

Start with:
```bash
docker-compose up -d
```

## API Usage

### Base URL
```
http://localhost:8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-06-15T23:51:40.542Z",
  "collection": {
    "name": "rag_docs",
    "count": 1250,
    "status": "healthy"
  },
  "cache": {
    "hits": 342,
    "misses": 58,
    "hit_rate": "85.50%",
    "size": 120
  }
}
```

### 1. Retrieve Documents (Hybrid Retrieval)

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What is machine learning?",
    "k": 3,
    "rerank_k": 50,
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1
  }'
```

Response:
```json
{
  "documents": [
    "Machine learning is a subset of artificial intelligence...",
    "ML algorithms can learn from data without being explicitly programmed...",
    "Deep learning is a powerful technique in machine learning..."
  ],
  "scores": [0.95, 0.88, 0.82],
  "ids": ["doc_001", "doc_002", "doc_003"],
  "count": 3,
  "timestamp": "2026-06-15T23:51:40.542Z"
}
```

### 2. Dense-Only Retrieval (Faster)

```bash
curl -X POST http://localhost:8000/retrieve/dense \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What is machine learning?",
    "k": 5
  }'
```

### 3. Generate Answer

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What is machine learning?",
    "context": [
      "Machine learning is a subset of artificial intelligence...",
      "ML algorithms can learn from data without being explicitly programmed..."
    ],
    "chat_history": []
  }'
```

### 4. Complete RAG Pipeline (One Call)

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What is machine learning?",
    "k": 3,
    "rerank_k": 50,
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1
  }'
```

Response:
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions...",
  "retrieved_documents": 3,
  "sources": ["doc_001", "doc_002", "doc_003"],
  "scores": [0.95, 0.88, 0.82],
  "timestamp": "2026-06-15T23:51:40.542Z"
}
```

### 5. Get Statistics

```bash
curl http://localhost:8000/stats \
  -H "X-API-Key: your-api-key"
```

## Configuration

### config.yaml Settings

```yaml
# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  file: "logs/rag_pipeline.log"
  max_bytes: 10485760  # 10 MB

# Model Configuration
models:
  embedding:
    name: "all-mpnet-base-v2"
    device: "auto"  # auto, cuda, mps, cpu
  cross_encoder:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: "auto"
  llm:
    name: "llama3"
    endpoint: "http://localhost:11434"

# Retrieval Weights
retrieval:
  alpha: 0.6    # Cross-encoder weight
  beta: 0.3     # Dense similarity weight
  gamma: 0.1    # TF-IDF weight

# Caching
caching:
  enabled: true
  max_size: 1000  # Number of embeddings to cache
  ttl_seconds: 3600

# Rate Limiting
rate_limiting:
  enabled: true
  requests_per_minute: 60
  requests_per_hour: 1000
```

## Performance Optimization

### 1. GPU Acceleration

The application auto-detects and uses:
- **NVIDIA CUDA** on Windows/Linux
- **Apple Metal (MPS)** on Mac
- **CPU fallback** if no GPU available

Check detection:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### 2. Enable GPU for Ollama

```bash
# Windows
set OLLAMA_GPU=cuda
ollama serve --gpu-layers 100

# Mac
export OLLAMA_GPU=metal
ollama serve

# Linux (NVIDIA)
export OLLAMA_GPU=cuda
ollama serve --gpu-layers 100
```

### 3. Embedding Caching

Caching is enabled by default (1000 embeddings). This significantly speeds up repeated queries.

Check cache hit rate:
```bash
curl http://localhost:8000/stats
```

### 4. Optimize Retrieval Weights

Tune `alpha`, `beta`, `gamma` in `config.yaml`:
- **High alpha (0.7+)**: Better accuracy, slower (needs cross-encoder)
- **High beta (0.5+)**: Fast, good for semantic similarity
- **High gamma (0.2+)**: Favor keyword matches

### 5. Reduce Candidate Pool

Lower `rerank_k` for faster reranking:
```json
{
  "rerank_k": 20,  # Instead of 50
  "k": 3
}
```

## Monitoring & Logging

### View Logs

```bash
# Real-time logs
tail -f logs/rag_pipeline.log

# Last 100 lines
tail -100 logs/rag_pipeline.log

# Filter by level
grep "ERROR" logs/rag_pipeline.log
```

### Log Levels

- **DEBUG**: Detailed information (slow queries, caching)
- **INFO**: General information (model loading, API calls)
- **WARNING**: Warnings (cross-encoder failed, using fallback)
- **ERROR**: Errors (retrieval failed, LLM error)

### Prometheus Metrics (Optional)

For production monitoring, add `prometheus-client`:

```python
from prometheus_client import Counter, Histogram

# In main.py
api_calls = Counter('rag_api_calls', 'Total API calls', ['endpoint'])
retrieval_time = Histogram('rag_retrieval_seconds', 'Retrieval latency')
```

## Authentication & Security

### Enable API Key Authentication

1. Set in `config.yaml`:
```yaml
security:
  api_key_enabled: true
  api_key: ${RAG_API_KEY}  # Set via environment
```

2. Set environment variable:
```bash
export RAG_API_KEY="sk-very-secret-key"
```

3. Include in requests:
```bash
curl -H "X-API-Key: sk-very-secret-key" http://localhost:8000/health
```

### Rate Limiting

Default limits:
- `/health`: 100 requests/minute
- `/retrieve`: 60 requests/minute
- `/generate`: 30 requests/minute
- `/rag/query`: 30 requests/minute

Adjust in `config.yaml` and FastAPI decorators.

### CORS Configuration

Modify `config.yaml`:
```yaml
api:
  cors_origins:
    - "https://yourfrontend.com"
    - "https://app.example.com"
```

## Testing

### Unit Tests

```bash
pytest tests/ -v
```

### Load Testing (with Apache Bench)

```bash
# Install
apt-get install apache2-utils

# Test /health endpoint
ab -n 1000 -c 10 http://localhost:8000/health

# Test retrieve endpoint
ab -p request.json -T application/json -n 100 -c 5 http://localhost:8000/retrieve
```

### Integration Tests

```python
# tests/test_api.py
import requests

def test_health():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200

def test_retrieve():
    payload = {"query": "test", "k": 3}
    resp = requests.post("http://localhost:8000/retrieve", json=payload)
    assert resp.status_code == 200
    assert "documents" in resp.json()
```

## Troubleshooting

### 1. "Model not found" Error

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve --gpu-layers 100
```

### 2. GPU Not Detected

```python
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

### 3. Slow Retrieval

- Check embedding cache hit rate: `curl http://localhost:8000/stats`
- Reduce `rerank_k` in request
- Disable cross-encoder: use `/retrieve/dense` instead
- Enable GPU for Ollama

### 4. Out of Memory

- Reduce `rerank_k` (candidate pool)
- Use smaller embedding model
- Use CPU instead of GPU (paradoxically faster for small datasets)

## Deployment Checklist

- [ ] Install dependencies: `pip install -r requirements_prod.txt`
- [ ] Pull LLM model: `ollama pull llama3`
- [ ] Configure `config.yaml` with production settings
- [ ] Enable API key authentication
- [ ] Set up logging to persistent storage
- [ ] Configure CORS for your frontend
- [ ] Set rate limits appropriately
- [ ] Test health endpoint: `curl http://localhost:8000/health`
- [ ] Run load tests
- [ ] Set up monitoring (logs, metrics)
- [ ] Deploy with Docker Compose or Kubernetes
- [ ] Enable GPU for Ollama if available

## Performance Benchmarks

Typical latencies (GPU-accelerated):
- Embedding generation: 50-200ms
- Reranking (50 candidates): 200-500ms
- LLM generation: 2-5 seconds (depends on context length)
- **Total RAG query**: 3-6 seconds

Without GPU (CPU-only):
- Embedding: 500ms-2s
- Reranking: 1-3s
- LLM: 5-15s
- **Total**: 8-20s

## Support & Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Chroma DB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Last Updated**: June 2026  
**Version**: 1.0.0 (Production)
