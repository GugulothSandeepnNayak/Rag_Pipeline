# Local Streamlit App

This folder contains the local development version of the RAG chatbot using Streamlit.

## Run locally

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ingest documents into Chroma:

```bash
python ingestion.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the app in your browser:

```text
http://localhost:8501
```

## Files

- `app.py` — Streamlit UI
- `rag_pipeline.py` — local retrieval and generation logic
- `ingestion.py` — document ingestion into Chroma
- `evaluation.py` — evaluation utilities
- `guardrails.py` — guardrail checks
- `build_index.py` — optional FAISS index builder
- `requirements.txt` — local dependencies
- `data/`, `chroma_db/`, `logs/` — local runtime data folders
