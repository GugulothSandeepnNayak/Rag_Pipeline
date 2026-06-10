# Project Documentation

This document describes each script in the repository, the key functions they provide, why they exist, and how they relate to one another.

---

## Quick overview
- `app.py`: Streamlit UI and orchestrator for queries, retrieval, generation, and evaluation.
- `rag_pipeline.py`: Retrieval + generation logic (Chroma client, embedding model, LLM call).
- `rag_pipeline.py`: Retrieval + generation logic (Chroma client, embedding model, LLM call). Now includes cross-encoder reranking and a TF-IDF hybrid signal, plus a `dense_retrieve()` helper.
- `ingestion.py`: Document reading, chunking, embedding, and storing into Chroma.
- `evaluation.py`: A set of metrics (BERT-like similarity, BLEU, precision, LLM-based rating).
- `build_index.py`: Script to build a FAISS index from precomputed chunks (offline indexing example).
- `guardrails.py`: Lightweight safety / validation helpers.
- `requirements.txt`: Python dependencies.

---

## app.py
Purpose: Provide a Streamlit front-end to interact with the RAG system.

Key behavior and functions:
- Initializes Streamlit session state to store chat `messages`.
- Accepts user input via `st.chat_input()`.
- Calls `retrieve(query)` from `rag_pipeline` to get relevant context chunks and scores.
- Calls `generate_answer()` from `rag_pipeline` to produce a response using an LLM (Ollama).
- Optionally computes evaluation metrics by calling `evaluate()` from `evaluation.py` when an expected answer is provided.
- Displays the top-k retrieved chunks and all retrieved sources; shows evaluation results and LLM scoring errors.

How/Why:
- `app.py` is the UX layer: it glues retrieval, LLM generation, and evaluation into a simple chat interface. It is intentionally lightweight and calls into the other modules for heavy lifting.

Relation to other scripts:
- `app.py` -> calls `rag_pipeline.retrieve()` and `rag_pipeline.generate_answer()`
- `app.py` -> calls `evaluation.evaluate()` for optional scoring

Run:
```bash
source .venv/bin/activate
streamlit run app.py
```

---

## rag_pipeline.py
Purpose: Handle vector retrieval and assemble prompts for the LLM.

Key functions:
- `retrieve(query, k=3)`
  - Encodes `query` using a `SentenceTransformer` model.
  - Queries the Chroma collection and returns `documents`, `scores`, and `ids`.
 - `retrieve(query, k=3, rerank_k=50, alpha=0.6, beta=0.3, gamma=0.1)`
   - Performs hybrid retrieval: fetches a larger candidate pool (dense retrieval), computes TF-IDF similarity, optionally applies a cross-encoder reranker, then combines signals (weights `alpha`/`beta`/`gamma`) to produce final ranked results.
 - `dense_retrieve(query, k=3)`
   - Simple dense-only retrieval wrapper used as a baseline for benchmarking.
- `generate_answer(chat_history, query, context_chunks)`
  - Builds a prompt combining conversation history and the `context_chunks` (joined text).
  - Calls `ollama.chat(...)` to obtain an LLM response and returns the text.

How/Why:
- This module keeps retrieval and generation together so `app.py` only needs to call two functions for RAG. The retrieval function abstracts the vector DB (Chroma) usage; `generate_answer` centralizes the prompt format used for the LLM.

Notes:
- The code uses a persistent Chroma client configured to `persist_directory` so that ingestion and the app share the same DB.
- Ollama is the chosen local model backend; ensure the Ollama daemon and the requested model (e.g., `llama3`) are installed and running.
- The retrieval pipeline now supports a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for high-quality re-ranking if available; it falls back gracefully if the cross-encoder is not present.

Relation to other scripts:
- `ingestion.py` writes into the same Chroma collection that `rag_pipeline.py` reads.
- `app.py` relies on `retrieve()` and `generate_answer()` provided here.

Tuning and benchmark:
- The Streamlit UI exposes `rerank_k`, final `k`, and weights `alpha/beta/gamma` so you can tune reranking behaviour live.
- Use `dense_retrieve()` as a baseline when measuring recall/precision for reranked results.

---

## ingestion.py
Purpose: Load documents from `data/`, chunk them, embed them, and store them in Chroma.

Key functions:
- `read_file(path)`
  - Supports PDF via `pypdf.PdfReader` and plain text files. Returns extracted text.
- `chunk_text(text, chunk_size=500)`
  - Splits raw text into sentence-based chunks approximately `chunk_size` characters long.
- `ingest_documents(folder='data')`
  - Walks files in `data/`, reads and chunks each, generates ids for each chunk, encodes chunks with `SentenceTransformer`, and stores them in Chroma in batches.

How/Why:
- Heavy documents must be split into smaller retrieval granularity (chunks). This module prepares embeddings and persists them to the vector DB so retrieval is fast at query time.

Important details:
- Batching is used to avoid memory spikes when encoding many chunks.
- The Chroma client is configured with a persistent `persist_directory` so the Streamlit app and ingestion process share the same DB files.

Run:
```bash
source .venv/bin/activate
python ingestion.py
```

---

## evaluation.py
Purpose: Offer automated evaluation metrics for model outputs.

Key functions:
- `bert_score_like(pred, ref)` — encodes `pred` and `ref` with `sentence-transformers` and returns cosine similarity.
- `bleu_score(pred, ref, max_n=4)` — lightweight cumulative BLEU implementation with brevity penalty.
- `precision(pred, ref)` — token-level precision (overlap / predicted tokens).
- `llm_score(pred, ref, model_name='llama3')` — ask a local LLM (via Ollama) to rate the prediction vs reference on a 0..1 scale.
- `evaluate(pred, ref, use_llm=False, llm_model='llama3')` — wrapper returning all metrics as a dictionary. If `use_llm` is enabled, attempts LLM scoring and reports errors in the result.

How/Why:
- These metrics are intended to give quick feedback (semantic similarity, token overlap, and an optional LLM-based judgement) for single examples or small batches. They are intentionally lightweight to avoid heavy external deps for BLEU.

Relation to other scripts:
- `app.py` calls `evaluate()` when an expected answer is entered in the UI to display scores and LLM evaluation.

---

## build_index.py
Purpose: Offline example of building a FAISS index from an `index/chunks.json` file.

Key behavior:
- Loads pre-saved chunks from `index/chunks.json`.
- Encodes them with a `SentenceTransformer` and writes a FAISS index to `index/faiss.index`.

How/Why:
- This script is an example of how to construct a FAISS index for nearest-neighbor search independent of Chroma. Useful if you want an on-disk, efficient similarity index without Chroma.

Relation to other scripts:
- Not directly used by `app.py` in the current flow. It's a utility for alternate indexing or experiments.

Run:
```bash
source .venv/bin/activate
python build_index.py
```

---

## guardrails.py
Purpose: Small helpers for content safety and simple validation.

Key functions:
- `is_toxic(query)` — checks whether `query` contains words in `BAD_WORDS`.
- `low_confidence(scores, threshold=1.5)` — quick thresholding helper (used for scoring heuristics).
- `validate_answer(answer, context)` — simple grounding check: returns True if any sentence in `answer` appears verbatim in `context`.

How/Why:
- These helpers are intentionally minimal and meant as examples/starting points — you should replace or extend them with stronger moderation/grounding logic for production.

Relation to other scripts:
- Could be called by `app.py` or `rag_pipeline.generate_answer()` to decide whether to show, suppress, or flag generated answers.

---

## Dependencies and Environment
- Python virtual environment (project uses `.venv`).
- Key packages are in `requirements.txt`: `streamlit`, `chromadb`, `sentence-transformers`, `ollama`, `pypdf`, `faiss-cpu`, etc.
- Ollama: the project uses `ollama` to call local models. The Ollama daemon and the chosen model (e.g. `llama3`) must be installed and running for generation and LLM evaluation to work.
- Chroma: We configure Chroma with `persist_directory` so ingestion and the app share the same vector store files under `./chroma_db`.

---

## How data flows (high level)
1. `ingestion.py` reads files from `data/`, chunks and embeds them, and stores them in Chroma (`./chroma_db`).
2. `app.py` receives user queries, calls `rag_pipeline.retrieve()` to fetch top-k chunks from Chroma, and passes those chunks to `rag_pipeline.generate_answer()` to compose the prompt.
3. `rag_pipeline.generate_answer()` calls the local LLM (Ollama) and returns the model output.
4. `app.py` optionally calls `evaluation.evaluate()` to score the output against an expected answer entered by the user.

---

## Troubleshooting notes
- If the UI shows empty chunks, ensure both ingestion and the app are configured to use the same Chroma `persist_directory` (this repo uses `./chroma_db`). Re-run `python ingestion.py` if needed.
- Ollama errors: check the Ollama daemon and model names. Some `ollama` client versions have slightly different call signatures — `evaluation.py` includes fallbacks.
- Large memory usage when encoding: ingestion uses batching to avoid spikes; reduce `batch_size` in `ingest_documents()` if you hit memory limits.

---

## Execution Flow & Retrieval Tuning (call chain)

1. User submits a query in the UI (`app.py`): the submit handler in the form captures `query`, `expected`, and flags like `use_llm_eval`.
2. `app.py` calls `retrieve()` in `rag_pipeline.py` (or `dense_retrieve()` when rerank is disabled).
   - `retrieve()` call sequence:
     - encode `query` with `SentenceTransformer` (`all-mpnet-base-v2`).
     - query Chroma collection to get an initial set of dense candidates (embedding nearest neighbors).
     - compute TF‑IDF similarity between the `query` and candidate chunks (lexical signal).
     - if cross-encoder reranker is enabled, run the cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to compute pairwise semantic scores for `rerank_k` candidates.
     - normalize signals and compute final score = `alpha * cross_score + beta * dense_score + gamma * tfidf_score`.
     - sort by final score and return `documents, scores, ids` limited to `final_k`.
3. `app.py` calls `generate_answer()` in `rag_pipeline.py` with `chat_history`, `query`, and the returned `documents`.
   - `generate_answer()` composes the prompt (history + retrieved context) and calls the LLM backend (`ollama.chat(...)`) to produce the final text.
4. Post-generation checks in `app.py`:
   - `guardrails.semantic_grounding_score(answer, contexts)` computes a per-answer semantic grounding score by encoding answer sentences and contexts and taking the maximum cosine similarity.
   - `guardrails.low_confidence(scores)` flags low retrieval confidence.
5. Optional evaluation: `app.py` calls `evaluate()` from `evaluation.py` to compute `bert_sim`, `bleu`, `precision`, and optionally `llm_score` (if `use_llm_eval` is set).
6. UI displays the assistant's text, the top-k chunks (with `ids` and the `scores` returned by `retrieve()`), plus the grounding score and evaluation metrics.

How the retrieval & reranking settings are used and how they affect output
- `use_rerank` (toggle):
  - Off: `dense_retrieve()` returns nearest neighbors by vector similarity only — faster, lower CPU.
  - On: `retrieve()` expands to a candidate pool and reranks — higher recall & typically better final relevance but higher cost.
- `rerank_k` (candidate pool size): larger values increase recall (the chance a relevant passage is in the pool) but increase compute (more cross-encoder calls and TF‑IDF work). Typical default: 50.
- `final_k` (results to return): how many chunks are returned and passed into the LLM prompt. Larger `final_k` gives the LLM more evidence (can reduce hallucination) but increases prompt size and cost — keep it small (2–4) unless you need many citations.
- `alpha` / `beta` / `gamma` (weights): control the blend of signals used for the final rank.
  - Increasing `alpha` gives more weight to the cross-encoder (semantic reranker) and often improves ranking quality when the cross-encoder is robust.
  - Increasing `beta` favors the original dense embedding ordering (good when embeddings are well aligned with queries).
  - Increasing `gamma` favors lexical overlap (useful for exact-match or named-entity queries).
  - The three weights are most interpretable when they sum to 1.0; you can tune them to trade precision vs recall.
- Practical trade-offs:
  - Latency vs Quality: enable reranking and increase `rerank_k` for quality; disable reranking for faster responses.
  - Hallucination risk: adding more high-quality, grounded chunks (`final_k`) and increasing `alpha` can reduce hallucination, but too many low-relevance chunks can confuse the model.
  - Cost: cross-encoder reranking is the most expensive step; TF‑IDF is cheap; dense retrieval cost depends on embedding model and DB.

What score is shown for the Top‑K chunks
- The `score` shown next to each chunk in the UI is the final combined score returned by `retrieve()` — the weighted blend of cross-encoder output, dense cosine similarity, and TF‑IDF lexical similarity (as controlled by `alpha`, `beta`, `gamma`).
- Additionally, the UI shows a semantic grounding score computed by `guardrails.semantic_grounding_score()` which measures how well the model's answer sentences semantically match the retrieved chunks (this is separate from the retrieval `score`).

Evaluation scores used / meaning
- `bert_sim`: sentence-transformer cosine similarity between the final answer and the expected reference (higher = closer semantic match). Used in `evaluate()` and in benchmarking for semantic match decisions.
- `bleu`: n-gram BLEU approximation (surface-level overlap).
- `precision`: token-overlap precision between predicted and reference tokens.
- `llm_score`: optional human-like score from the local LLM (Ollama) when `use_llm_eval` is enabled.

Quick suggestions
- Start with `use_rerank=True`, `rerank_k=50`, `final_k=3`, `alpha=0.6`, `beta=0.3`, `gamma=0.1` (these are the UI defaults).
- If the system returns irrelevant chunks, raise `rerank_k` or `alpha`; if it returns lexically exact but semantically wrong passages, increase `gamma` or add more precise query formulations.
- Use the grounding score threshold (default 0.78) to decide whether to show a grounding warning — tune this threshold if you need a stricter or looser definition of 'grounded'.

If you want, I can add a small diagram or a sequence-of-calls box to the top of `DOCUMENTATION.md` linking to specific functions by line. Would you like a diagram or the sequence as clickable file links? 

---

## Scaling, Deployment, and Maintenance

This section describes practical steps to scale the RAG system, deploy it reliably, and maintain it when new data arrives.

1. Containerization & reproducible environments
  - Provide a `Dockerfile` for the app (Python dependencies + model artifacts where possible). Bundle a lightweight runtime for the embedding model; rely on hosted model caches (HF_TOKEN) or pre-download weights during build where license permits.
  - Example components: `web` (Streamlit), `worker` (ingestion/batch embedding), `vector-db` (Chroma files or managed Chroma endpoint), optional `llm` connector (Ollama host) and `storage` (S3/volume).

2. Deployment patterns
  - Small / local: `docker-compose` with volumes for `./chroma_db` and a bound `./data` for ingestion. Keep Ollama on the host or in its own container.
  - Production: Kubernetes deployment with persistent volumes (PV/PVC) for Chroma `persist_directory`, HorizontalPodAutoscaler for stateless web workers, and a separate job queue (Celery/RQ/Kafka) for ingest and embedding tasks.
  - Use a managed vector DB (or host Chroma centrally) for multi-instance scale; ensure consistent `persist_directory` or a single central store.

3. Ingestion & incremental updates
  - Keep ingestion idempotent: compute deterministic chunk ids (e.g., `filename_page_chunkidx`) and upsert into Chroma so repeated runs don't duplicate.
  - Implement a lightweight watcher or scheduled job that:
    - Detects new files (S3 notifications, filesystem events) → downloads to staging → runs chunking/embedding in a worker → upserts embeddings to Chroma.
  - Batch embeddings to conserve memory and speed GPU throughput; tune `batch_size` per hardware.
  - Record ingestion metadata (timestamp, source file, version) alongside chunk metadata to support audits and selective reindexing.

4. Reindexing and schema migrations
  - Maintain versioned chunk metadata (e.g., `schema_version`) and provide a reindex script to rebuild embeddings if you change embedding model or chunking logic.
  - For safe migrations: run reindex into a new collection (or path) and swap pointers/aliases to avoid downtime.

5. Monitoring, testing & metrics
  - Monitor key metrics: ingestion rate, vector DB size, query latency, LLM latency/cost, retrieval recall on a small probe benchmark.
  - Add automated smoke tests on deploy: run a few sample queries and validate expected top-k hits or minimal grounding scores.
  - Persist benchmark runs and compare over time to detect regressions after model or pipeline changes.

6. Backups, restore & retention
  - Periodically snapshot the Chroma `persist_directory` to object storage (S3/Glacier). Keep ingestion metadata and raw documents in a separate backup store.
  - Keep a retention policy for old data and a documented restore procedure.

7. Security & access control
  - Restrict access to the Chroma persistence volume and any LLM endpoints. Use network policies, IAM, and service accounts.
  - Sanitize and scan ingested documents for PII if your use case requires it. Maintain an allow/denylist for file types and reasonable file size limits.

8. Cost & resource planning
  - Embedding and cross-encoder models can be CPU/GPU intensive. Use GPU workers for large-scale reindexing; reserve smaller CPU instances for web-serving.
  - Consider managed vector DBs for simpler ops vs running Chroma yourself.

9. CI/CD & automation
  - Automate model and dependency updates in CI; run end-to-end tests that include small sample ingestion and retrieval runs.
  - Use feature-flagged rollouts when changing reranking defaults or embedding models; rollback quickly if benchmark metrics degrade.

10. Operational tips for future data
  - Validate new documents before ingesting: file size, parseable text, language detection.
  - Maintain a changelog of ingestion batches and a mapping from source documents to chunk ids (helpful for provenance and deletion).
  - Provide an admin UI or CLI to remove or reprocess specific documents/chunks if source data is corrected.

If you want, I can generate:
- a `Dockerfile` and `docker-compose.yml` for local staging,
- a Kubernetes manifest (Deployment + PV + HPA) example,
- an ingestion-worker script with S3-trigger handling, or
- a reindexing script that writes to a new Chroma collection and swaps.
Which artifact would you like first?

---

## Cloud deployment (Azure)

This subsection maps the project's components to Azure services and gives concrete patterns for ingestion, embedding, vector storage, retrieval, and hosting.

- **Storage (raw data):** Azure Blob Storage (container `raw-docs`).
- **Eventing:** Azure Event Grid (blob-created events) → Azure Function or Event Grid Subscription.
- **Ingestion worker:** Azure Functions (Python) or Containerized worker on Azure Container Apps/AKS. Worker responsibilities:
  - Download blob, extract text (pypdf / Form Recognizer for scanned PDFs), chunk text deterministically, compute chunk ids.
  - Call embedding endpoint in batches.
  - Upsert vectors + metadata to the vector index.
  - Record ingestion metadata to Cosmos DB / Azure Table Storage for provenance.
- **Embeddings provider:**
  - Azure OpenAI embeddings (managed, low ops) — recommended for production for simplicity and scale.
  - Alternative: host SentenceTransformers on Azure ML or AKS (for full control / offline use).
- **Vector store options:**
  - Azure Cognitive Search with vector fields (managed vector search) — lowest ops and integrates with Azure security.
  - Self-hosted Chroma/Qdrant on AKS with PVs (gives full control; needs maintenance and backups).
- **Reranker / cross-encoder:** host as an Azure ML online endpoint or containerized microservice (AKS / Container Apps). Run only on candidate pools to control cost.
- **LLM / generation:**
  - Azure OpenAI (recommended) for generation and optional LLM-based evaluation.
  - If self-hosting, deploy model on Azure ML or GPU VM and expose via secure endpoint.
- **App hosting:** Azure Container Apps / App Service for Containers (for ease) or AKS for larger deployments.
- **Secrets & identity:** Azure Key Vault + Managed Identities for secure credentials and service access.
- **Monitoring & logging:** Azure Monitor + Application Insights + Log Analytics; track ingestion rates, query latency, embedding throughput, and benchmark metrics.

Patterns and Tips
- **Event-driven incremental ingestion:** Blob uploaded → Event Grid triggers Function → Function processes and upserts. Use deterministic chunk ids and metadata so ingestion is idempotent.
- **Batching & scaling:** Use container workers for heavy embedding jobs (batch) and Functions for lightweight or near-real-time ingestion. Use AKS with autoscaling or Container Apps with scale rules for workers.
- **Zero-downtime reindexing:** Write reindexed vectors to a new Cognitive Search index (or new Chroma collection), run validation/benchmark, then swap an alias or update application config to point to the new index.
- **Security:** use private endpoints, VNet integration, and Key Vault-backed secrets. Restrict app and worker identities to minimal privileges.
- **Cost control:** offload heavy cross-encoder reranks to on-demand endpoints; cache embeddings and popular query results; use batch embedding for initial loads.

Quick Azure artifacts I can scaffold for you
- `azure-function/` Python blob-trigger template that: downloads blobs, extracts text, chunks, calls Azure OpenAI embeddings, and upserts to Cognitive Search (with example index schema).
- `docker/` + `docker-compose.yml` for local testing that simulates Blob Storage (local files), an ingestion worker, and a local vector DB (Chroma) for staging.
- ARM/Terraform snippet to provision Blob Storage, Cognitive Search (vector-capable index), Key Vault, and App Service.

Tell me which artifact you'd like first and I will scaffold it (I recommend the `azure-function` blob-trigger + Cognitive Search upsert example to get incremental ingestion working on Azure).

---

## Azure Databricks: scalable ingestion, embedding, and reindexing

Use Databricks when you need scalable, reliable ETL and embedding pipelines for large or frequently-updated corpora. Databricks is best used for batch processing, heavy embedding workloads, Delta Lake storage, and ML lifecycle management (MLflow). Keep a low-latency vector search service (Azure Cognitive Search, Qdrant, Chroma on AKS) for serving queries.

Recommended pattern:
- Raw files land in ADLS Gen2 / Blob Storage (`raw-docs`).
- Databricks Auto Loader / cloudFiles detects new files and ingests into a landing Delta table (`landing_files`).
- A scheduled or streaming Databricks notebook performs:
  1. Read new/updated files from the landing table using watermarking.
  2. Extract text (use Python libraries or call Form Recognizer for scanned PDFs).
  3. Chunk text deterministically and compute chunk metadata (chunk_id, doc_id, offset, page).
  4. Write chunks to a Delta table `delta.chunks` with `ingested_at`, `batch_id`, and `schema_version`.
  5. Batch embeddings per partition (mapPartitions) calling Azure OpenAI embeddings API or local encoder on GPU nodes.
  6. Upsert embeddings to the vector store (Cognitive Search / Qdrant / Chroma) and record the `vector_id` in `delta.chunks`.

Advantages:
- Delta tables provide ACID guarantees and make reprocessing and auditing straightforward.
- MLflow integration lets you register embedding / reranker models and track versions used to encode vectors.

Practical code pointers to implement in Databricks:
- Delta schema for `chunks` (example columns): `chunk_id STRING, doc_id STRING, offset INT, text STRING, metadata MAP, embedding_model STRING, vector_id STRING, ingested_at TIMESTAMP`
- Use `writeStream` with `cloudFiles` for near-real-time ingestion or a scheduled job for batch.
- Use `mapInPandas` or `mapPartitions` to call the embedding API in batches and write back embeddings/metadata.

Best practices and operational notes
- Deterministic chunk IDs: derive from `sha256(file_path + offset)` or `filename_page_chunkidx` so re-runs are idempotent and you can upsert rather than duplicate.
- Watermarking: use a `last_modified` watermark or Auto Loader state to only process new/changed files.
- Schema versioning: store `chunking_version` and `embedding_model_version` to know which chunks must be reindexed after model changes.
- Upserts: when writing to the vector DB, upsert by `chunk_id` / `vector_id` to avoid duplicates.
- Provenance: store ingestion metadata (source blob path, batch_id, ingested_at, model versions) in `ingestion_log` Delta table for audits and selective reprocessing.
- Reprocessing: to reindex only changed files or to switch embedding models, create a reindex job that selects chunks by `embedding_model_version != desired_version` and re-embeds them into a new index/collection.

Testing and validation
- Use a small probe dataset and run validation notebooks that perform a set of sample queries and assert expected `top_k` hits or minimum grounding/bert_sim scores.
- Persist benchmark results in a `delta.benchmarks` table to detect regressions across reindex or model changes.

Security & secrets
- Use Databricks secrets backed by Azure Key Vault for API keys (Azure OpenAI key, Cognitive Search admin key).
- Limit access to Delta tables using Unity Catalog and enforce least privilege for notebooks and jobs.

Next-step scaffolding I can produce
- A Databricks notebook that demonstrates Auto Loader → chunking → Delta `chunks` write → batch embeddings (Azure OpenAI) → upsert to Cognitive Search (Python + PySpark code samples).
- Example Delta table DDL and a small test dataset with a scheduled Job JSON for Databricks.

If you'd like, I can scaffold the Databricks notebook and the Delta schema next — should I proceed with the notebook that writes to Azure Cognitive Search, or would you prefer a notebook that writes vectors to a self-hosted Qdrant/Chroma endpoint? 

---

## Retrieval tuning options

This project exposes several retrieval and reranking options in the Streamlit UI. Use these to trade off latency vs. quality and to tune the signal blending used to rank candidate chunks.

- **Rerank (Enable re-ranking)**: When enabled, the system first fetches a larger candidate pool using dense retrieval and then reranks that pool using a cross-encoder (semantic re-scoring) and a TF‑IDF lexical signal. This typically improves final relevance at the cost of additional compute.
- **`rerank_k` (candidate pool size / pool size)**: The number of initial dense candidates to fetch before reranking. Larger values increase the chance of including a relevant passage but increase embedding and/or cross-encoder computation. Typical default: `50`.
- **`k` / `final_k` (results to return)**: The number of final chunks returned to the UI and passed to the LLM as context. Smaller values reduce prompt size and cost; typical default: `3`.
- **Weights `alpha`, `beta`, `gamma`**: These are the blending weights applied to the different signals when computing the final score for reranking.
  - `alpha`: cross-encoder weight (semantic re-ranker). Higher values give more influence to the cross-encoder.
  - `beta`: dense-vector similarity weight (original embedding cosine). Higher values preserve the dense retrieval ordering.
  - `gamma`: TF‑IDF lexical similarity weight. Adds a lexical match signal to prefer passages that lexically overlap the query.
  - The weights should sum to 1.0 for an interpretable convex combination; defaults in the UI are `alpha=0.6`, `beta=0.3`, `gamma=0.1`.
- **`use_rerank`**: UI toggle to enable or disable reranking. Disabling it falls back to `dense_retrieve()` (dense-only baseline).
- **`bench_threshold`**: Used by the benchmark CSV runner as the minimum `bert_sim` value considered a semantic match. Typical default: `0.7`.
- **`use_llm_eval`**: When checked, the evaluation step will call the local LLM (Ollama) to produce an additional human-like score for `pred` vs `ref`. This requires Ollama running and may be slower.

Tips and tradeoffs:
- If latency is the priority, reduce `rerank_k` and/or disable reranking; increase `beta` to rely more on dense similarity.
- If quality is the priority and you have CPU for cross-encoder reranking, increase `rerank_k` and set a higher `alpha` to let the cross-encoder reorder candidates.
- Use `final_k=3` as a reasonable default for providing concise context to most LLMs; increase to `5+` for larger context windows or when more evidence is needed.
- For benchmarks, keep `rerank_k` stable between runs and only vary weights to compare ranking strategies fairly.

If you'd like, I can add example recommended presets (fast / balanced / high-quality) to the UI and documentation.

---

If you'd like, I can:
- Add inline code links to each function (pointing to file/line anchors).
- Generate a `README.md` with quick start commands and screenshots.
- Produce a small `docs/` folder with per-module examples and tests.

File created: `DOCUMENTATION.md` in project root.
