 # Documentation RAG Chatbot

This repository is a retrieval-augmented generation (RAG) demo that ingests documentation into a vector store and serves a Streamlit chat UI with retrieval, reranking, evaluation, and simple guardrails.

## What is included
- `app.py` – Streamlit chat UI and orchestration.
- `ingestion.py` – Read files, chunk text, embed and upsert to Chroma.
- `rag_pipeline.py` – Retrieval, TF‑IDF hybrid scoring, optional cross‑encoder reranking, and generation.
- `evaluation.py` – BERT-sim, BLEU, precision and optional LLM scoring helpers.
- `guardrails.py` – Toxicity checks and semantic grounding score.
- `build_index.py` – (optional) indexing helpers.
- `DOCUMENTATION.md` – Detailed design, tuning and deployment notes.

## Quick start (local)
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ingest documents into the vector DB (Chroma):

```bash
source .venv/bin/activate
python ingestion.py
```

Chroma persists into `./chroma_db` by default.

4. Run the Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501 to use the chat UI.

## Configuration and environment
- The code uses `sentence-transformers` embeddings by default (`all-mpnet-base-v2`).
- Ollama is optional — if you use local Ollama for generation or LLM scoring, ensure the daemon is running and configure any endpoints in `rag_pipeline.py` / `evaluation.py`.
- Adjust retrieval parameters in the UI: `rerank_k`, `final_k`, and weights `alpha/beta/gamma` for hybrid scoring.

## Evaluation
- Use `evaluation.py` to compute `bert_sim`, BLEU, and precision scores. LLM-based scoring is optional and will fall back gracefully if not available.

## Guardrails
- `guardrails.py` applies simple regex-based toxicity checks and a semantic grounding score threshold (BERT-sim). Tune the grounding threshold in `app.py` or `guardrails.py`.

## Deployment notes
- See `DOCUMENTATION.md` for guidance on Docker, Kubernetes, Azure, and Databricks deployment patterns.
- For production: protect your models and vector DB, add auth, use managed vector DBs or durable object storage, and add stronger moderation APIs.

## Pushing to GitHub
Create a repository named `Documentation-RAG-chatbot` and push your local repo. Example commands:

```bash
git branch -M main
git remote add origin https://github.com/<USERNAME>/Documentation-RAG-chatbot.git
git push -u origin main
```

## Troubleshooting
- If embeddings fail with memory/segfault, re-run ingestion with smaller batches (see `ingestion.py`).
- If `gh` CLI is missing, install via Homebrew: `brew install gh`.

## License
This workspace has no license file. Add one if you plan to publish.

---
Made for quick local experiments with RAG workflows. See `DOCUMENTATION.md` for details.
