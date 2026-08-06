"""
routes.py — FastAPI API Router for Chatbot Endpoints.

Exposes REST API routes:
  - POST /query       : Main chat endpoint (handles streaming and standard JSON responses).
  - GET  /status      : System health check.
  - GET  /schema      : Database schema metadata text.
  - POST /clear-cache : Flushes backend caches & reloads prompts/guardrails.

All pipeline execution logic lives in src/domain/orchestrator/pipeline_orchestrator.py.
All caching logic lives in src/core/cache/cache_manager.py.
All MCP session logic lives in src/domain/mcp/session_manager.py.
"""

# ── MODULE TAG: FastAPI Endpoint Router ──
from fastapi import APIRouter, HTTPException
from api.models import ChatRequest
from orchestrator.pipeline_orchestrator import process_query_pipeline
from core.cache.cache_manager import clear_all_caches
from db.schema import get_schema
from core.config.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.post("/query")
async def handle_query(req: ChatRequest, stream: bool = False):
    """
    Main chat endpoint.

    Accepts a natural-language question and returns:
        sql:            The SQL query executed (if any)
        columns:        Column names from the query result
        rows:           Data rows from the query result
        natural_answer: Formatted human-readable answer
        error:          Error message if something went wrong
        attempts:       Number of agent loop steps taken
        agent_name:     Display name of the agent that answered

    Query param:
        stream (bool, default False): Returns StreamingResponse of token chunks when True.
    """
    return await process_query_pipeline(req, stream=stream)


@router.get("/status")
async def health_check():
    """Return a simple OK response to confirm the service is running."""
    return {"ok": True, "message": "service running"}


@router.get("/schema")
async def schema_endpoint():
    """Return the full database schema as a formatted text string."""
    try:
        return {"schema": get_schema()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/clear-cache")
async def clear_cache_endpoint():
    """Clear all backend caches: response TTLCache, semantic cache, schema cache, and registry cache."""
    try:
        clear_all_caches()
        log.info("All backend caches, guardrails, and prompt configurations reloaded via API request.")
        return {"ok": True, "message": "All backend caches and guardrail patterns cleared & reloaded successfully."}
    except Exception as e:
        log.error(f"Failed to clear backend caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
