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
    """Main query execution endpoint for natural language to SQL conversion.

    Args:
        req (ChatRequest): Incoming chat request payload containing user_question and session context.
        stream (bool, optional): Streaming response flag. Defaults to False.

    Returns:
        Union[dict, StreamingResponse]: Structured query result payload or StreamingResponse token stream.
    """
    return await process_query_pipeline(req, stream=stream)


@router.get("/status")
async def health_check():
    """Return system health metrics including MCP pool state, LLM throttle, and deduplication stats.

    Returns:
        dict: Health status object containing pool availability and performance metrics.
    """
    from mcp_service.mcp_session_pool import mcp_pool
    from core.llm.llm_throttle import get_throttle_status
    from core.request_deduplicator import get_dedup_stats

    return {
        "ok": True,
        "message": "service running",
        "mcp_pool": {
            "pool_size": mcp_pool.size,
            "available_sessions": mcp_pool.available,
            "is_ready": mcp_pool.is_ready,
        },
        "gemini_throttle": get_throttle_status(),
        "deduplicator": get_dedup_stats(),
    }


@router.get("/schema")
async def schema_endpoint():
    """Return the database schema metadata text.

    Returns:
        dict: Object containing formatted schema metadata string.

    Raises:
        HTTPException: If schema retrieval fails with status 400.
    """
    try:
        return {"schema": get_schema()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/clear-cache")
async def clear_cache_endpoint():
    """Flush all backend caches and reload system prompt configurations and guardrails.

    Returns:
        dict: Confirmation response object.

    Raises:
        HTTPException: If cache clearing fails with status 500.
    """
    try:
        clear_all_caches()
        log.info("All backend caches, guardrails, and prompt configurations reloaded via API request.")
        return {"ok": True, "message": "All backend caches and guardrail patterns cleared & reloaded successfully."}
    except Exception as e:
        log.error(f"Failed to clear backend caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
