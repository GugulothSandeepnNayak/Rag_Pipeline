# Production-ready RAG API

This folder contains the production-ready FastAPI version of the RAG pipeline.

## Run production

1. Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_prod.txt
```

2. Configure `config.yaml` as needed.

3. Run the API server:

```bash
python main.py
```

4. Or run with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

5. Open the health endpoint:

```bash
http://localhost:8000/health
```

## Docker

Build and run the production container:

```bash
docker build -t rag-pipeline .
docker run -p 8000:8000 -e RAG_API_KEY="your-secret-key" -v %CD%/chroma_db:/app/chroma_db -v %CD%/logs:/app/logs rag-pipeline
```

## Files

- `main.py` — FastAPI server
- `rag_pipeline_prod.py` — production retrieval/generation pipeline
- `Dockerfile` — production container image
- `docker-compose.yml` — optional multi-container stack
- `config.yaml` — production configuration
- `requirements_prod.txt` — production dependencies
