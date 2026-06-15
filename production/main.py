"""
Production-grade RAG API using FastAPI
- Rate limiting
- Request validation
- Authentication
- Error handling
- Structured logging
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from typing import List, Optional
import yaml
import os
from datetime import datetime

# Import production RAG pipeline
from rag_pipeline_prod import (
    retrieve, dense_retrieve, generate_answer,
    get_cache_stats, get_collection_info,
    logger, config
)

# ===== CONFIG =====
app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0"
)

# ===== RATE LIMITING =====
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "retry_after": 60}
    )

# ===== CORS =====
cors_config = config.get("api", {}).get("cors_origins", ["http://localhost:3000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== AUTHENTICATION =====
def verify_api_key(request: Request) -> str:
    """Verify API key if enabled."""
    security_config = config.get("security", {})
    if not security_config.get("api_key_enabled", False):
        return "default"
    
    api_key = request.headers.get("X-API-Key")
    expected_key = os.getenv("RAG_API_KEY", security_config.get("api_key"))
    
    if not api_key or api_key != expected_key:
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

# ===== REQUEST/RESPONSE MODELS =====
class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    k: int = Field(3, ge=1, le=20, description="Number of results")
    rerank_k: int = Field(50, ge=5, le=500, description="Candidate pool size")
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Cross-encoder weight")
    beta: float = Field(0.3, ge=0.0, le=1.0, description="Dense similarity weight")
    gamma: float = Field(0.1, ge=0.0, le=1.0, description="TF-IDF weight")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('alpha', 'beta', 'gamma')
    def weights_sum_check(cls, v, values):
        # Weights should sum to approximately 1
        if 'alpha' in values and 'beta' in values:
            total = values.get('alpha', 0) + values.get('beta', 0) + v
            if total > 1.1:
                logger.warning(f"Weights sum to {total:.2f}, may not normalize properly")
        return v

class DenseRetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(3, ge=1, le=20)

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=5000)

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    context: List[str] = Field(..., min_items=1, max_items=20)
    chat_history: List[ChatMessage] = Field(default_factory=list)

class RetrievalResponse(BaseModel):
    documents: List[str]
    scores: List[float]
    ids: List[Optional[str]]
    count: int
    timestamp: str

class GenerateResponse(BaseModel):
    answer: str
    query: str
    context_count: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    collection: dict
    cache: dict
    api_version: str

# ===== ROUTES =====

@app.get("/health", response_model=HealthResponse)
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        collection=get_collection_info(),
        cache=get_cache_stats(),
        api_version="1.0.0"
    )

@app.post("/retrieve", response_model=RetrievalResponse)
@limiter.limit("60/minute")
async def retrieve_endpoint(
    req: RetrievalRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieve documents using hybrid retrieval with reranking.
    
    Returns top-k documents ranked by cross-encoder, dense similarity, and TF-IDF.
    """
    try:
        logger.info(f"[{api_key}] Retrieve request: query={req.query[:50]}..., k={req.k}")
        
        documents, scores, ids = retrieve(
            query=req.query,
            k=req.k,
            rerank_k=req.rerank_k,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma
        )
        
        return RetrievalResponse(
            documents=documents,
            scores=scores,
            ids=ids,
            count=len(documents),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/retrieve/dense", response_model=RetrievalResponse)
@limiter.limit("60/minute")
async def dense_retrieve_endpoint(
    req: DenseRetrievalRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Simple dense-only retrieval (no reranking).
    
    Faster but less accurate than hybrid retrieval.
    """
    try:
        logger.info(f"[{api_key}] Dense retrieve request: query={req.query[:50]}..., k={req.k}")
        
        documents, scores, ids = dense_retrieve(
            query=req.query,
            k=req.k
        )
        
        return RetrievalResponse(
            documents=documents,
            scores=scores,
            ids=ids,
            count=len(documents),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Dense retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dense retrieval failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("30/minute")
async def generate_endpoint(
    req: GenerateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate an answer using LLM with provided context.
    
    Optionally include chat history for multi-turn conversations.
    """
    try:
        logger.info(f"[{api_key}] Generate request: query={req.query[:50]}..., context_count={len(req.context)}")
        
        # Convert chat_history to dict format
        chat_history = [{"role": msg.role, "content": msg.content} for msg in req.chat_history]
        
        answer = generate_answer(
            chat_history=chat_history,
            query=req.query,
            context_chunks=req.context
        )
        
        return GenerateResponse(
            answer=answer,
            query=req.query,
            context_count=len(req.context),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/rag/query")
@limiter.limit("30/minute")
async def rag_query_endpoint(
    req: RetrievalRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Complete RAG pipeline: retrieve documents and generate answer in one call.
    """
    try:
        logger.info(f"[{api_key}] RAG query: {req.query[:50]}...")
        
        # Retrieve
        documents, scores, ids = retrieve(
            query=req.query,
            k=req.k,
            rerank_k=req.rerank_k,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma
        )
        
        # Generate
        if not documents:
            return JSONResponse(
                status_code=404,
                content={"detail": "No documents found for this query"}
            )
        
        answer = generate_answer(
            chat_history=[],
            query=req.query,
            context_chunks=documents
        )
        
        return {
            "query": req.query,
            "answer": answer,
            "retrieved_documents": len(documents),
            "sources": ids,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/stats")
@limiter.limit("100/minute")
async def stats_endpoint(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Get pipeline statistics (cache, collection info)."""
    return {
        "cache": get_cache_stats(),
        "collection": get_collection_info(),
        "timestamp": datetime.now().isoformat()
    }

# ===== ERROR HANDLERS =====
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ===== STARTUP/SHUTDOWN =====
@app.on_event("startup")
async def startup_event():
    """Called when server starts."""
    logger.info("RAG API server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Called when server shuts down."""
    logger.info("RAG API server shutting down")

if __name__ == "__main__":
    import uvicorn
    
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    workers = api_config.get("workers", 4)
    
    logger.info(f"Starting RAG API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False
    )
