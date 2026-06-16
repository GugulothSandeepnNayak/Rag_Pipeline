import chromadb
from chromadb.config import Settings
import ollama
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch

# Detect GPU availability
def get_device():
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("GPU not available, using CPU")
        return "cpu"

device = get_device()

# Load models with GPU support if available
model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Use the same persistent directory as ingestion so the app can read the
# previously ingested documents across processes.
client = chromadb.Client(Settings(is_persistent=True, persist_directory="./chroma_db", allow_reset=True))

# Prefer existing collection when available; create if missing to avoid
# chromadb.errors.NotFoundError at import time (Streamlit reloads often).
try:
    collection = client.get_collection(name="rag_docs")
except Exception:
    # fallback to creating the collection if it doesn't exist
    collection = client.get_or_create_collection(name="rag_docs")


    
try:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
except Exception:
    cross_encoder = None
def retrieve(query, k=3, rerank_k=50, alpha=0.6, beta=0.3, gamma=0.1):
    """Retrieve top-k documents for `query` with cross-encoder reranking and
    a simple hybrid TF-IDF + dense retrieval combination.

    Parameters:
    - query: str
    - k: final number of results to return
    - rerank_k: number of candidates to fetch from the vector DB before reranking
    - alpha/beta/gamma: weights for cross-encoder, dense-sim, and TF-IDF respectively
    """
    if rerank_k < k:
        rerank_k = max(k, rerank_k)

    # dense retrieval (larger candidate set)
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=rerank_k
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # some Chroma versions return ids as 'ids'
    ids = results.get("ids")
    if ids:
        ids = ids[0]
    else:
        ids = [None] * len(documents)

    if len(documents) == 0:
        return [], [], []

    # Cross-encoder scoring
    cross_scores = None
    if cross_encoder is not None:
        pairs = [[query, doc] for doc in documents]
        try:
            cross_scores = np.array(cross_encoder.predict(pairs))
        except Exception:
            cross_scores = None

    # TF-IDF similarity (query vs each candidate)
    try:
        vect = TfidfVectorizer().fit([query] + documents)
        q_vec = vect.transform([query])
        d_vecs = vect.transform(documents)
        # cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf_sims = cosine_similarity(q_vec, d_vecs)[0]
    except Exception:
        tfidf_sims = np.zeros(len(documents))

    # convert distances to similarity-like scores (smaller distance -> larger sim)
    try:
        dist_arr = np.array(distances, dtype=float)
        dist_sims = 1.0 / (1.0 + dist_arr)
    except Exception:
        dist_sims = np.zeros(len(documents))

    # Normalize each score to [0,1]
    def norm(x):
        x = np.array(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    cross_norm = norm(cross_scores) if cross_scores is not None else np.zeros(len(documents))
    tfidf_norm = norm(tfidf_sims)
    dist_norm = norm(dist_sims)

    final_scores = alpha * cross_norm + beta * dist_norm + gamma * tfidf_norm

    # pick top-k indices
    top_idx = np.argsort(-final_scores)[:k]

    top_docs = [documents[i] for i in top_idx]
    top_scores = [float(final_scores[i]) for i in top_idx]
    top_ids = [ids[i] for i in top_idx]

    return top_docs, top_scores, top_ids


def dense_retrieve(query, k=3):
    """Simple dense-only retrieval using Chroma (no reranking)."""
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ids = results.get("ids")
    if ids:
        ids = ids[0]
    else:
        ids = [None] * len(documents)

    # convert distances to similarity-like score
    try:
        import numpy as _np

        dist_arr = _np.array(distances, dtype=float)
        sims = (1.0 / (1.0 + dist_arr)).tolist()
    except Exception:
        sims = [0.0] * len(documents)

    return documents, sims, ids


def generate_answer(chat_history, query, context_chunks):
    context = "\n\n".join(context_chunks)

    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a helpful AI assistant.

Conversation so far:
{history_text}

Use the context below to answer the question.

Context:
{context}

User: {query}
Assistant:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
