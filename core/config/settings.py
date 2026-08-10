"""
settings.py — All environment variables and shared constants for the chatbot.

Every other module imports from here instead of reading os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ── LLM & Gemini Config ──────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", os.getenv("MODEL_NAME", "gemini-2.5-flash"))
MODEL_NAME: str = GEMINI_MODEL
ROUTER_MODEL_NAME: str = os.getenv("ROUTER_MODEL_NAME", GEMINI_MODEL)
REPHRASER_MODEL_NAME: str = os.getenv("REPHRASER_MODEL_NAME", GEMINI_MODEL)
MAX_RETRIES: int = 3
DB_TIMEOUT: float = 10.0

# ── Database connection ────────────────────────────────────────────────────────
DB_DIALECT:  str = os.getenv("DB_DIALECT",  "mysql")
DB_USER:     str = os.getenv("DB_USER",     "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_HOST:     str = os.getenv("DB_HOST",     "localhost")
DB_PORT:     str = os.getenv("DB_PORT",     "3306")
DB_NAME:     str = os.getenv("DB_NAME",     "WMS_DB")

# ── WMS API Endpoint ────────────────────────────────────────────────────────────
WMS_API_URL: str = os.getenv("WMS_API_URL", "")
WMS_API_TIMEOUT: int = int(os.getenv("WMS_API_TIMEOUT", "30"))

# ── Domain table pools ─────────────────────────────────────────────────────────
# WMS domain — single unified table pool used by WMS_AGENT
WMS_TABLES: frozenset = frozenset({
    "ITEM", "SKUITEM", "SULOCATION", "LOCATION", "PICKLIST",
    "PICKLISTITEM", "PICKLISTVIEW", "GRN", "FGMODEL",
    "ITEMLOCACNMAP", "FGTRANSACTION", "SUIDACTIVITYLOG",
    "WAREHOUSE", "user",
})

# ── RAG / embedding ────────────────────────────────────────────────────────────
GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
DEFAULT_EMBED_MODEL: str = GEMINI_EMBED_MODEL
VECTOR_RAG_THRESHOLD: float = 0.70  # Cosine similarity cutoff for RAG table vector search
OLLAMA_THRESHOLD: float = VECTOR_RAG_THRESHOLD  # Backwards compatibility alias
MOCK_DIM: int = 512  # Hash-bucket vector dimension
REGISTRY_TTL: int = 3600 * 6  

# ── Semantic cache ─────────────────────────────────────────────────────────────
SEMANTIC_CACHE_THRESHOLD: float = 0.99  # cosine sim cutoff for semantic cache hits
SEMANTIC_CACHE_TTL:       int   = int(os.getenv("SEMANTIC_CACHE_TTL", "50"))   # seconds before a semantic cache entry expires
SEMANTIC_CACHE_MAX:       int   = 200   # max entries before oldest are evicted

# ── Caching ────────────────────────────────────────────────────────────────────
SCHEMA_CACHE_TTL:  int = 600   # seconds before schema cache expires
INTENT_CACHE_TTL:  int = int(os.getenv("INTENT_CACHE_TTL", "30"))   # seconds before intent cache expires
API_CACHE_TTL:     int = int(os.getenv("API_CACHE_TTL", "30"))   # seconds before query response cache expires
SQL_CACHE_TTL:     int = int(os.getenv("SQL_CACHE_TTL", "30"))   # seconds before SQL result cache expires (mcp server)

# ── Prompts directory ──────────────────────────────────────────────────────────
PROMPTS_DIR: str = os.path.join(PROJECT_ROOT, "prompts")


# ── Agent loop tuning ──────────────────────────────────────────────────────────
AGENT_MAX_STEPS:      int = 5     # maximum reasoning iterations per SQL agent request
HISTORY_WINDOW:       int = 6     # number of past chat messages included in context
