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
from fastapi.responses import HTMLResponse, JSONResponse
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

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Pipeline Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-soft: #f0f3f6;
      --text: #15191f;
      --muted: #68717d;
      --line: #d9dee5;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --accent-soft: #dff5f1;
      --warn: #b45309;
      --error: #b91c1c;
      --shadow: 0 18px 45px rgba(18, 28, 45, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.88), rgba(246,247,249,0.96)),
        radial-gradient(circle at 16% 0%, rgba(15,118,110,0.13), transparent 28%),
        radial-gradient(circle at 86% 12%, rgba(48,86,211,0.10), transparent 30%),
        var(--bg);
      color: var(--text);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button,
    input,
    textarea {
      font: inherit;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      width: min(1440px, calc(100vw - 32px));
      min-height: calc(100vh - 32px);
      margin: 16px auto;
    }

    .sidebar,
    .workspace {
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(217,222,229,0.9);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      min-height: 0;
      border-radius: 8px;
      overflow: hidden;
    }

    .brand {
      padding: 22px 22px 16px;
      border-bottom: 1px solid var(--line);
    }

    .brand h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .brand p {
      margin: 8px 0 0;
      color: var(--muted);
    }

    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }

    .metric {
      min-width: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }

    .metric strong {
      display: block;
      margin-top: 3px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 18px;
    }

    .controls {
      overflow: auto;
      padding: 16px;
    }

    .section-title {
      margin: 0 0 12px;
      font-size: 13px;
      font-weight: 700;
      color: #303843;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .field {
      margin-bottom: 14px;
    }

    .field label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
      color: #303843;
      font-weight: 600;
    }

    .field label output {
      color: var(--muted);
      font-weight: 500;
    }

    input[type="text"],
    input[type="password"],
    input[type="number"],
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      outline: none;
      padding: 10px 12px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    textarea {
      resize: vertical;
    }

    input:focus,
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15,118,110,0.14);
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }

    .segmented button {
      border: 0;
      border-radius: 6px;
      padding: 9px 10px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
    }

    .segmented button.active {
      background: var(--panel);
      color: var(--text);
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    .presets {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 8px 0 18px;
    }

    .secondary {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: #303843;
      cursor: pointer;
      padding: 9px 10px;
    }

    .secondary:hover {
      border-color: #b8c0cc;
      background: #f8fafc;
    }

    .workspace {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      min-height: 0;
      border-radius: 8px;
      overflow: hidden;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
    }

    .topbar h2 {
      margin: 0;
      font-size: 18px;
    }

    .topbar .links {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .topbar a,
    .ghost {
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: #303843;
      text-decoration: none;
      padding: 8px 11px;
      cursor: pointer;
    }

    .chat {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 35%);
      min-height: 0;
    }

    .messages {
      min-width: 0;
      overflow: auto;
      padding: 20px;
      background: linear-gradient(180deg, rgba(248,250,252,0.82), rgba(255,255,255,0.9));
    }

    .empty {
      max-width: 680px;
      margin: 10vh auto 0;
      text-align: center;
      color: var(--muted);
    }

    .empty h3 {
      margin: 0 0 8px;
      color: var(--text);
      font-size: 28px;
      line-height: 1.15;
    }

    .chips {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: #303843;
      cursor: pointer;
      padding: 8px 12px;
    }

    .message {
      max-width: 860px;
      margin: 0 0 16px;
      display: flex;
      gap: 10px;
    }

    .message.user {
      margin-left: auto;
      flex-direction: row-reverse;
    }

    .avatar {
      flex: 0 0 34px;
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-weight: 800;
    }

    .message.user .avatar {
      background: #e7ecf7;
      color: #34456f;
    }

    .bubble {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: #fff;
      white-space: pre-wrap;
    }

    .message.user .bubble {
      background: #ecfdf5;
      border-color: #bbf7d0;
    }

    .bubble.error {
      border-color: #fecaca;
      background: #fff1f2;
      color: var(--error);
    }

    .sidepanel {
      min-width: 0;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: #fbfcfe;
      padding: 16px;
    }

    .sidepanel h3 {
      margin: 0 0 12px;
      font-size: 16px;
    }

    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin-bottom: 10px;
    }

    .source header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .source p {
      margin: 0;
      color: #303843;
      display: -webkit-box;
      -webkit-line-clamp: 8;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: rgba(255,255,255,0.94);
    }

    .composer textarea {
      min-height: 52px;
      max-height: 160px;
    }

    .primary {
      min-width: 112px;
      min-height: 52px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      padding: 12px 18px;
    }

    .primary:hover {
      background: var(--accent-strong);
    }

    .primary:disabled,
    .secondary:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .notice {
      margin-top: 12px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fff7ed;
      color: var(--warn);
      padding: 10px 12px;
    }

    .muted {
      color: var(--muted);
    }

    @media (max-width: 980px) {
      .shell {
        grid-template-columns: 1fr;
        min-height: auto;
      }

      .chat {
        grid-template-columns: 1fr;
      }

      .sidepanel {
        border-left: 0;
        border-top: 1px solid var(--line);
        max-height: 360px;
      }
    }

    @media (max-width: 640px) {
      .shell {
        width: 100%;
        margin: 0;
      }

      .sidebar,
      .workspace {
        border-radius: 0;
        border-left: 0;
        border-right: 0;
      }

      .topbar,
      .composer {
        grid-template-columns: 1fr;
      }

      .composer {
        display: block;
      }

      .primary {
        width: 100%;
        margin-top: 10px;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <aside class="sidebar">
      <section class="brand">
        <h1>RAG Pipeline Console</h1>
        <p>Ask questions, tune retrieval, inspect sources, and monitor the production API from one place.</p>
      </section>

      <section class="status-grid" aria-label="Pipeline status">
        <div class="metric">
          <span>Status</span>
          <strong id="statusText">Checking</strong>
        </div>
        <div class="metric">
          <span>Documents</span>
          <strong id="docCount">-</strong>
        </div>
        <div class="metric">
          <span>Cache Hit Rate</span>
          <strong id="cacheRate">-</strong>
        </div>
        <div class="metric">
          <span>Last Latency</span>
          <strong id="latency">-</strong>
        </div>
      </section>

      <section class="controls">
        <p class="section-title">Retrieval Mode</p>
        <div class="field">
          <div class="segmented" role="tablist" aria-label="Retrieval mode">
            <button type="button" id="modeRag" class="active">Full RAG</button>
            <button type="button" id="modeRetrieve">Retrieve Only</button>
          </div>
        </div>

        <p class="section-title">Tuning</p>
        <div class="presets">
          <button type="button" class="secondary" data-preset="fast">Fast</button>
          <button type="button" class="secondary" data-preset="balanced">Balanced</button>
          <button type="button" class="secondary" data-preset="quality">Quality</button>
        </div>

        <div class="field">
          <label for="k">Results <output id="kOut">3</output></label>
          <input id="k" type="range" min="1" max="20" step="1" value="3" />
        </div>

        <div class="field">
          <label for="rerankK">Candidate Pool <output id="rerankOut">50</output></label>
          <input id="rerankK" type="range" min="5" max="500" step="5" value="50" />
        </div>

        <div class="field">
          <label for="alpha">Cross Encoder <output id="alphaOut">0.60</output></label>
          <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.6" />
        </div>

        <div class="field">
          <label for="beta">Dense Similarity <output id="betaOut">0.30</output></label>
          <input id="beta" type="range" min="0" max="1" step="0.05" value="0.3" />
        </div>

        <div class="field">
          <label for="gamma">TF-IDF <output id="gammaOut">0.10</output></label>
          <input id="gamma" type="range" min="0" max="1" step="0.05" value="0.1" />
        </div>

        <p class="section-title">Security</p>
        <div class="field">
          <label for="apiKey">API Key</label>
          <input id="apiKey" type="password" autocomplete="off" placeholder="Optional if enabled" />
        </div>
        <button type="button" id="refreshHealth" class="secondary">Refresh Status</button>
      </section>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <div>
          <h2>Chat</h2>
          <div class="muted" id="subtitle">Connected to production FastAPI</div>
        </div>
        <nav class="links" aria-label="API links">
          <a href="/docs">API Docs</a>
          <a href="/health">Health</a>
          <button type="button" id="clearChat" class="ghost">Clear</button>
        </nav>
      </header>

      <section class="chat">
        <div id="messages" class="messages" aria-live="polite">
          <div class="empty" id="emptyState">
            <h3>Ask your production RAG pipeline</h3>
            <p>Use the chat box below, then inspect retrieved chunks and scores on the right.</p>
            <div class="chips">
              <button type="button" class="chip">What is generative AI?</button>
              <button type="button" class="chip">Summarize the uploaded document.</button>
              <button type="button" class="chip">What are the key benefits of LLMs?</button>
            </div>
          </div>
        </div>

        <aside class="sidepanel">
          <h3>Sources</h3>
          <div id="sourceList" class="muted">Sources will appear after a query.</div>
          <div id="emptyDbNotice" class="notice" hidden>
            The production collection is empty. Point production at a populated Chroma DB or ingest documents into production before expecting grounded answers.
          </div>
        </aside>
      </section>

      <form id="composer" class="composer">
        <textarea id="query" rows="2" placeholder="Ask a question about your documents..." required></textarea>
        <button id="send" class="primary" type="submit">Send</button>
      </form>
    </section>
  </main>

  <script>
    const state = {
      mode: "rag",
      busy: false,
      messages: []
    };

    const $ = (id) => document.getElementById(id);
    const messagesEl = $("messages");
    const emptyState = $("emptyState");
    const sourceList = $("sourceList");
    const emptyDbNotice = $("emptyDbNotice");

    const controls = {
      k: $("k"),
      rerankK: $("rerankK"),
      alpha: $("alpha"),
      beta: $("beta"),
      gamma: $("gamma")
    };

    function headers() {
      const h = { "Content-Type": "application/json" };
      const key = $("apiKey").value.trim();
      if (key) h["X-API-Key"] = key;
      return h;
    }

    function payload(query) {
      return {
        query,
        k: Number(controls.k.value),
        rerank_k: Number(controls.rerankK.value),
        alpha: Number(controls.alpha.value),
        beta: Number(controls.beta.value),
        gamma: Number(controls.gamma.value)
      };
    }

    function setBusy(next) {
      state.busy = next;
      $("send").disabled = next;
      $("send").textContent = next ? "Running" : "Send";
    }

    function syncOutputs() {
      $("kOut").textContent = controls.k.value;
      $("rerankOut").textContent = controls.rerankK.value;
      $("alphaOut").textContent = Number(controls.alpha.value).toFixed(2);
      $("betaOut").textContent = Number(controls.beta.value).toFixed(2);
      $("gammaOut").textContent = Number(controls.gamma.value).toFixed(2);
    }

    function setMode(mode) {
      state.mode = mode;
      $("modeRag").classList.toggle("active", mode === "rag");
      $("modeRetrieve").classList.toggle("active", mode === "retrieve");
      $("subtitle").textContent = mode === "rag"
        ? "Full RAG: retrieve context and generate an answer"
        : "Retrieve Only: inspect matching chunks without LLM generation";
    }

    function applyPreset(name) {
      const presets = {
        fast: { k: 3, rerankK: 15, alpha: 0, beta: 1, gamma: 0 },
        balanced: { k: 3, rerankK: 50, alpha: 0.6, beta: 0.3, gamma: 0.1 },
        quality: { k: 5, rerankK: 100, alpha: 0.7, beta: 0.2, gamma: 0.1 }
      };
      Object.entries(presets[name]).forEach(([key, value]) => {
        controls[key].value = value;
      });
      syncOutputs();
    }

    function addMessage(role, content, tone = "") {
      emptyState.hidden = true;
      const wrap = document.createElement("article");
      wrap.className = `message ${role}`;

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "You" : "AI";

      const bubble = document.createElement("div");
      bubble.className = `bubble ${tone}`;
      bubble.textContent = content;

      wrap.append(avatar, bubble);
      messagesEl.appendChild(wrap);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderSources(docs = [], scores = [], ids = []) {
      sourceList.innerHTML = "";
      if (!docs.length) {
        sourceList.className = "muted";
        sourceList.textContent = "No sources returned.";
        return;
      }
      sourceList.className = "";
      docs.forEach((doc, index) => {
        const card = document.createElement("section");
        card.className = "source";
        const score = Number(scores[index]);
        card.innerHTML = `
          <header>
            <strong>Chunk ${index + 1}</strong>
            <span>${ids[index] || "no id"}${Number.isFinite(score) ? ` | ${score.toFixed(3)}` : ""}</span>
          </header>
          <p></p>
        `;
        card.querySelector("p").textContent = doc || "";
        sourceList.appendChild(card);
      });
    }

    async function refreshHealth() {
      try {
        const res = await fetch("/health", { headers: headers() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        $("statusText").textContent = data.status || "unknown";
        $("docCount").textContent = data.collection?.count ?? "-";
        $("cacheRate").textContent = data.cache?.hit_rate ?? "-";
        emptyDbNotice.hidden = Number(data.collection?.count || 0) !== 0;
      } catch (error) {
        $("statusText").textContent = "Offline";
        $("docCount").textContent = "-";
        $("cacheRate").textContent = "-";
      }
    }

    async function submitQuery(query) {
      if (state.busy) return;
      setBusy(true);
      addMessage("user", query);
      const started = performance.now();

      try {
        const endpoint = state.mode === "rag" ? "/rag/query" : "/retrieve";
        const res = await fetch(endpoint, {
          method: "POST",
          headers: headers(),
          body: JSON.stringify(payload(query))
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.detail || `Request failed with HTTP ${res.status}`);
        }

        const elapsed = performance.now() - started;
        $("latency").textContent = `${Math.round(elapsed)} ms`;

        if (state.mode === "rag") {
          addMessage("assistant", data.answer || "No answer returned.");
          renderSources(data.documents || data.context || [], data.scores || [], data.sources || []);
          if (!data.documents && data.sources) {
            sourceList.innerHTML = "";
            data.sources.forEach((id, index) => {
              const card = document.createElement("section");
              card.className = "source";
              card.innerHTML = `<header><strong>Source ${index + 1}</strong><span>${id || "no id"} | ${Number(data.scores?.[index] || 0).toFixed(3)}</span></header><p>Open Retrieve Only mode to inspect source text.</p>`;
              sourceList.appendChild(card);
            });
          }
        } else {
          addMessage("assistant", `Retrieved ${data.count || 0} document chunk(s).`);
          renderSources(data.documents || [], data.scores || [], data.ids || []);
        }

        await refreshHealth();
      } catch (error) {
        addMessage("assistant", error.message, "error");
      } finally {
        setBusy(false);
      }
    }

    $("composer").addEventListener("submit", (event) => {
      event.preventDefault();
      const q = $("query").value.trim();
      if (!q) return;
      $("query").value = "";
      submitQuery(q);
    });

    $("query").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        $("composer").requestSubmit();
      }
    });

    $("modeRag").addEventListener("click", () => setMode("rag"));
    $("modeRetrieve").addEventListener("click", () => setMode("retrieve"));
    $("refreshHealth").addEventListener("click", refreshHealth);
    $("clearChat").addEventListener("click", () => {
      messagesEl.querySelectorAll(".message").forEach((el) => el.remove());
      emptyState.hidden = false;
      renderSources();
      $("latency").textContent = "-";
    });

    document.querySelectorAll("[data-preset]").forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });

    document.querySelectorAll(".chip").forEach((button) => {
      button.addEventListener("click", () => {
        $("query").value = button.textContent;
        $("query").focus();
      });
    });

    Object.values(controls).forEach((control) => control.addEventListener("input", syncOutputs));
    syncOutputs();
    refreshHealth();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the production RAG chat console."""
    return HTMLResponse(INDEX_HTML)

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
            "documents": documents,
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
