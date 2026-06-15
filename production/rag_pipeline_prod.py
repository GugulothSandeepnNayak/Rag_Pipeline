import chromadb
from chromadb.config import Settings
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import logging
from functools import lru_cache
from typing import Tuple, List
import yaml
import os
from datetime import datetime

# ===== LOGGING SETUP =====
def setup_logging(config: dict):
    """Configure logging based on config settings."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_config.get("file", "logs/rag_pipeline.log")
    
    # Create logs directory if missing
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ===== CONFIG LOADING =====
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config("config.yaml")
logger = setup_logging(config)

logger.info("="*50)
logger.info("Initializing RAG Pipeline (Production)")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info("="*50)

# ===== DEVICE DETECTION =====
def get_device() -> str:
    """Auto-detect GPU: NVIDIA (CUDA), Apple (MPS), or fallback to CPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ CUDA GPU detected: {gpu_name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("✓ Metal GPU detected (Apple Silicon)")
        return "mps"
    else:
        logger.warning("⚠ No GPU available, using CPU")
        return "cpu"

device_config = config.get("models", {}).get("embedding", {}).get("device", "auto")
if device_config == "auto":
    device = get_device()
else:
    device = device_config
    logger.info(f"Using configured device: {device}")

# ===== MODEL LOADING =====
def load_embedding_model(model_name: str, device: str):
    """Load embedding model with error handling."""
    try:
        logger.info(f"Loading embedding model: {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"✓ Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError(f"Could not load embedding model: {e}")

def load_cross_encoder(model_name: str, device: str):
    """Load cross-encoder with error handling."""
    try:
        logger.info(f"Loading cross-encoder: {model_name} on {device}")
        cross_encoder = CrossEncoder(model_name, device=device)
        logger.info(f"✓ Cross-encoder loaded successfully")
        return cross_encoder
    except Exception as e:
        logger.warning(f"Failed to load cross-encoder: {e}. Reranking disabled.")
        return None

# Load models
embedding_config = config.get("models", {}).get("embedding", {})
cross_encoder_config = config.get("models", {}).get("cross_encoder", {})

model = load_embedding_model(embedding_config.get("name", "all-mpnet-base-v2"), device)
cross_encoder = load_cross_encoder(cross_encoder_config.get("name", "cross-encoder/ms-marco-MiniLM-L-6-v2"), device)

# ===== CHROMA DB SETUP =====
def init_chroma_collection(config: dict):
    """Initialize Chroma client and collection with error handling."""
    chroma_config = config.get("chroma", {})
    persist_dir = chroma_config.get("persist_directory", "./chroma_db")
    collection_name = chroma_config.get("collection_name", "rag_docs")
    
    try:
        logger.info(f"Initializing Chroma with persist_directory: {persist_dir}")
        client = chromadb.Client(Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            allow_reset=True
        ))
        
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"✓ Using existing collection: {collection_name}")
        except Exception:
            logger.info(f"Creating new collection: {collection_name}")
            collection = client.get_or_create_collection(name=collection_name)
        
        return client, collection
    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")
        raise RuntimeError(f"Could not initialize Chroma DB: {e}")

client, collection = init_chroma_collection(config)

# ===== CACHING =====
class EmbeddingCache:
    """Simple LRU cache for embeddings."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str) -> np.ndarray:
        if query in self.cache:
            self.hits += 1
            return self.cache[query]
        self.misses += 1
        return None
    
    def set(self, query: str, embedding: np.ndarray):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO fallback)
            self.cache.pop(next(iter(self.cache)))
        self.cache[query] = embedding
    
    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self.cache)
        }

cache_config = config.get("caching", {})
cache_enabled = cache_config.get("enabled", True)
embedding_cache = EmbeddingCache(max_size=cache_config.get("max_size", 1000)) if cache_enabled else None

# ===== RETRIEVAL FUNCTIONS =====
def retrieve(query: str, k: int = 3, rerank_k: int = 50, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1) -> Tuple[List[str], List[float], List[str]]:
    """
    Retrieve top-k documents with hybrid retrieval + reranking.
    
    Parameters:
    - query: Search query
    - k: Final results to return
    - rerank_k: Candidate pool size
    - alpha: Cross-encoder weight
    - beta: Dense-similarity weight
    - gamma: TF-IDF weight
    
    Returns: (documents, scores, ids)
    """
    logger.debug(f"Retrieving for query: {query[:100]}...")
    
    try:
        if rerank_k < k:
            rerank_k = max(k, rerank_k)
        
        # Get embedding (with caching)
        if embedding_cache:
            query_embedding = embedding_cache.get(query)
            if query_embedding is None:
                query_embedding = model.encode(query)
                embedding_cache.set(query, query_embedding)
            else:
                logger.debug("Using cached embedding")
        else:
            query_embedding = model.encode(query)
        
        # Dense retrieval
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=rerank_k
        )
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [None] * len(documents))
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        
        if len(documents) == 0:
            logger.warning(f"No documents found for query: {query}")
            return [], [], []
        
        logger.info(f"Retrieved {len(documents)} candidates")
        
        # Cross-encoder scoring
        cross_scores = None
        if cross_encoder is not None:
            try:
                pairs = [[query, doc] for doc in documents]
                cross_scores = np.array(cross_encoder.predict(pairs))
            except Exception as e:
                logger.warning(f"Cross-encoder scoring failed: {e}")
                cross_scores = None
        
        # TF-IDF similarity
        try:
            vect = TfidfVectorizer().fit([query] + documents)
            q_vec = vect.transform([query])
            d_vecs = vect.transform(documents)
            tfidf_sims = cosine_similarity(q_vec, d_vecs)[0]
        except Exception as e:
            logger.warning(f"TF-IDF computation failed: {e}")
            tfidf_sims = np.zeros(len(documents))
        
        # Convert distances to similarity
        try:
            dist_arr = np.array(distances, dtype=float)
            dist_sims = 1.0 / (1.0 + dist_arr)
        except Exception as e:
            logger.warning(f"Distance conversion failed: {e}")
            dist_sims = np.zeros(len(documents))
        
        # Normalization
        def norm(x):
            x = np.array(x, dtype=float)
            if len(x) == 0 or x.max() == x.min():
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())
        
        cross_norm = norm(cross_scores) if cross_scores is not None else np.zeros(len(documents))
        tfidf_norm = norm(tfidf_sims)
        dist_norm = norm(dist_sims)
        
        # Hybrid scoring
        final_scores = alpha * cross_norm + beta * dist_norm + gamma * tfidf_norm
        
        # Select top-k
        top_idx = np.argsort(-final_scores)[:k]
        top_docs = [documents[i] for i in top_idx]
        top_scores = [float(final_scores[i]) for i in top_idx]
        top_ids = [ids[i] if i < len(ids) else None for i in top_idx]
        
        logger.info(f"Retrieved {len(top_docs)} final results")
        return top_docs, top_scores, top_ids
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise

def dense_retrieve(query: str, k: int = 3) -> Tuple[List[str], List[float], List[str]]:
    """Simple dense-only retrieval (no reranking)."""
    logger.debug(f"Dense retrieval for query: {query[:100]}...")
    
    try:
        if embedding_cache:
            query_embedding = embedding_cache.get(query)
            if query_embedding is None:
                query_embedding = model.encode(query)
                embedding_cache.set(query, query_embedding)
        else:
            query_embedding = model.encode(query)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [None] * len(documents))
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        
        try:
            dist_arr = np.array(distances, dtype=float)
            sims = (1.0 / (1.0 + dist_arr)).tolist()
        except Exception as e:
            logger.warning(f"Distance conversion failed: {e}")
            sims = [0.0] * len(documents)
        
        logger.info(f"Dense retrieval returned {len(documents)} results")
        return documents, sims, ids
    
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}", exc_info=True)
        raise

# ===== ANSWER GENERATION =====
def generate_answer(chat_history: List[dict], query: str, context_chunks: List[str]) -> str:
    """Generate answer using LLM with retrieved context."""
    logger.debug(f"Generating answer for query: {query[:100]}...")
    
    try:
        context = "\n\n".join(context_chunks)
        
        history_text = ""
        for msg in chat_history:
            history_text += f"{msg['role']}: {msg['content']}\n"
        
        prompt = f"""You are a helpful AI assistant.

Conversation so far:
{history_text}

Use the context below to answer the question.

Context:
{context}

User: {query}
Assistant:"""
        
        llm_config = config.get("models", {}).get("llm", {})
        model_name = llm_config.get("name", "llama3")
        
        logger.debug(f"Calling LLM: {model_name}")
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.get("message", {}).get("content", "No response")
        logger.info(f"Answer generated ({len(answer)} chars)")
        return answer
    
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        raise RuntimeError(f"Could not generate answer: {e}")

# ===== INFO FUNCTIONS =====
def get_cache_stats() -> dict:
    """Get embedding cache statistics."""
    if embedding_cache:
        return embedding_cache.stats()
    return {"enabled": False}

def get_collection_info() -> dict:
    """Get collection metadata."""
    try:
        count = collection.count()
        return {
            "name": collection.name,
            "count": count,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Could not get collection info: {e}")
        return {"status": "error", "error": str(e)}

logger.info("RAG Pipeline initialization complete")
