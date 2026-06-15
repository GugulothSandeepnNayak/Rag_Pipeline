# Production RAG Pipeline - Quick Start Guide

## 🚀 Start in 5 Minutes

### Prerequisites
- Python 3.10+
- Ollama installed (https://ollama.ai)
- GPU optional (but recommended)

### 1. Install Dependencies

```bash
pip install -r requirements_prod.txt
```

### 2. Pull LLM Model

```bash
ollama pull llama3
```

Or use a smaller/faster model:
```bash
ollama pull orca-mini
```

### 3. Start API Server

```bash
# Option A: Simple (development)
python main.py

# Option B: Production (multi-worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will be available at: **http://localhost:8000**

### 4. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "collection": {"name": "rag_docs", "count": 1250},
  "cache": {"hits": 0, "misses": 0, "size": 0}
}
```

## 📝 Example API Calls

### Complete RAG Query (Retrieve + Generate)

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 3
  }'
```

### Just Retrieve Documents

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 5,
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1
  }'
```

### Just Generate Answer (from context)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the importance of machine learning",
    "context": [
      "Machine learning is transforming industries...",
      "AI is increasingly important for business..."
    ]
  }'
```

## 🐳 Run with Docker

```bash
# Start Ollama + API in containers
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop
docker-compose down
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Model names & device (GPU/CPU)
- API host/port and workers
- Rate limits
- Cache settings
- Logging level

Example configuration changes:

```yaml
# Use GPU
models:
  embedding:
    device: "cuda"  # or "mps" for Mac
  cross_encoder:
    device: "cuda"

# Faster retrieval (less accurate)
retrieval:
  alpha: 0.2    # Less reranking
  beta: 0.8     # More vector search
  gamma: 0.0    # No TF-IDF

# Increase rate limits
rate_limiting:
  requests_per_minute: 120
```

## 📊 Monitor Performance

Get cache statistics and collection info:

```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "cache": {
    "hits": 342,
    "misses": 58,
    "hit_rate": "85.50%",
    "size": 120
  },
  "collection": {
    "name": "rag_docs",
    "count": 1250
  }
}
```

## 🔐 Enable Authentication

For production, enable API key authentication:

1. Edit `config.yaml`:
```yaml
security:
  api_key_enabled: true
```

2. Set environment variable:
```bash
export RAG_API_KEY="sk-your-secret-key"
```

3. Include in API requests:
```bash
curl http://localhost:8000/health \
  -H "X-API-Key: sk-your-secret-key"
```

## 🚦 Check Logs

```bash
# View logs in real-time
tail -f logs/rag_pipeline.log

# Filter errors only
grep ERROR logs/rag_pipeline.log
```

## 🛠️ Troubleshooting

### "Model not found" Error
```bash
# Make sure Ollama is running
ollama serve --gpu-layers 100

# In another terminal:
ollama pull llama3
```

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Responses
- Lower `rerank_k` in requests: `"rerank_k": 20`
- Use `/retrieve/dense` endpoint (faster, less accurate)
- Enable GPU for Ollama: `OLLAMA_GPU=1`

## 📚 Full Documentation

See [PRODUCTION_README.md](PRODUCTION_README.md) for:
- Architecture overview
- Full API documentation
- Deployment options
- Performance tuning
- Security best practices
- Monitoring setup

## 🔗 Useful Links

- **FastAPI Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Ollama API**: http://localhost:11434/api/tags
- **Chroma Docs**: https://docs.trychroma.com/

---

**Ready to deploy!** 🎉
