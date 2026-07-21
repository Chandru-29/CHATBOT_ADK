"""
settings.py — All environment variables and shared constants for the chatbot.

Every other module imports from here instead of reading os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
MODEL_NAME:  str   = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
ROUTER_MODEL_NAME:    str = os.getenv("ROUTER_MODEL_NAME", "qwen2.5-coder:7b")
REPHRASER_MODEL_NAME: str = os.getenv("REPHRASER_MODEL_NAME", "qwen2.5-coder:7b")
FORMATTER_MODEL_NAME: str = os.getenv("FORMATTER_MODEL_NAME", "qwen2.5-coder:7b")
MAX_RETRIES: int   = 3
DB_TIMEOUT:  float = 10.0

# ── Database connection ────────────────────────────────────────────────────────
DB_DIALECT:  str = os.getenv("DB_DIALECT",  "mysql")
DB_USER:     str = os.getenv("DB_USER",     "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_HOST:     str = os.getenv("DB_HOST",     "localhost")
DB_PORT:     str = os.getenv("DB_PORT",     "3306")
DB_NAME:     str = os.getenv("DB_NAME",     "company_data")

# ── Domain table pools ─────────────────────────────────────────────────────────
# Used by the intent router to scope which tables each agent can query.
HR_TABLES:    frozenset = frozenset({"employees", "departments", "projects","attendance"})
SALES_TABLES: frozenset = frozenset({"customers", "orders","products"})

# ── RAG / embedding ────────────────────────────────────────────────────────────
DEFAULT_EMBED_MODEL: str   = "nomic-embed-text"
OLLAMA_THRESHOLD:    float = 0.53  # cosine sim cutoff for real Ollama embeddings
MOCK_THRESHOLD:      float = 0.10   # cosine sim cutoff for trigram mock vectors
MOCK_DIM:            int   = 512    # hash-bucket vector dimension
REGISTRY_TTL:        int   = 3600*6  

# ── Semantic cache ─────────────────────────────────────────────────────────────
SEMANTIC_CACHE_THRESHOLD: float = 0.92  # cosine sim cutoff for semantic cache hits
SEMANTIC_CACHE_TTL:       int   = 300   # seconds before a semantic cache entry expires
SEMANTIC_CACHE_MAX:       int   = 200   # max entries before oldest are evicted

# ── Caching ────────────────────────────────────────────────────────────────────
SCHEMA_CACHE_TTL:  int = 600   # seconds before schema cache expires
INTENT_CACHE_TTL:  int = 120   # seconds before intent cache expires
API_CACHE_TTL:     int = 120   # seconds before query response cache expires
SQL_CACHE_TTL:     int = 120   # seconds before SQL result cache expires (mcp server)

# ── Semantic cache (cosine-similarity question dedup) ─────────────────────────
SEMANTIC_CACHE_THRESHOLD: float = 0.92  # cosine similarity cutoff for a cache hit
SEMANTIC_CACHE_TTL:       int   = 300   # seconds before a semantic cache entry expires
SEMANTIC_CACHE_MAX:       int   = 200   # max number of (vector, result) pairs stored

# ── Prompts directory ──────────────────────────────────────────────────────────
PROMPTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
