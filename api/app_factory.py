"""
app_factory.py — Creates the FastAPI application and registers all route modules.

Import create_app() from here in main.py to get the configured app instance.
"""

# ── MODULE TAG: FastAPI App Factory & MCP Lifespan ──
import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config.logger import get_logger
from core.config.settings import PROJECT_ROOT
from db.engine import DB_URL

log = get_logger(__name__)

_mcp_state: dict = {
    "read":    None,
    "write":   None,
    "session": None,
    "cm_stdio": None,
    "cm_sess":  None,
}

_server_params: StdioServerParameters | None = None

# ── MCP Session Pool (replaces single session) ──────────────────────────────────
# The pool is imported here so it can be started/stopped in the lifespan handler.
from mcp_service.mcp_session_pool import mcp_pool  # noqa: E402


def build_server_params() -> StdioServerParameters:
    """Construct parameters to run FastMCP server script as a Python subprocess.

    Returns:
        StdioServerParameters: FastMCP server subprocess parameter configuration.
    """
    server_script = os.path.join(
        PROJECT_ROOT, "mcp_service", "server.py"
    )

    return StdioServerParameters(
        command=sys.executable,
        args=[server_script, DB_URL],
    )


async def _start_mcp() -> None:
    """Initialize and start the persistent MCP session pool."""
    global _server_params
    _server_params = build_server_params()
    try:
        await mcp_pool.start(_server_params)
        _mcp_state["session"] = None
        log.info(
            f"MCPSessionPool started with {mcp_pool.size} sessions "
            f"({mcp_pool.available} available)."
        )
    except Exception as e:
        log.error(f"MCPSessionPool startup failed: {e}")


async def _stop_mcp() -> None:
    """Shut down and terminate all sessions in the MCP session pool."""
    try:
        await mcp_pool.stop()
    except Exception as e:
        log.warning(f"MCPSessionPool stop warning: {e}")
    for key in ("read", "write", "session", "cm_stdio", "cm_sess"):
        _mcp_state[key] = None
    log.info("MCPSessionPool shut down.")


def get_mcp_session() -> None:
    """Legacy compatibility MCP session getter shim.

    Returns:
        None: Always returns None as sessions are managed via MCPSessionPool.
    """
    return None


async def restart_mcp() -> None:
    """Restart the full MCP session pool."""
    log.warning("MCP restart triggered — restarting session pool…")
    await _stop_mcp()
    await _start_mcp()


async def _daily_semantic_cache_flusher() -> None:
    """Background worker task that automatically flushes the ChromaDB semantic_cache collection every 24 hours."""
    log.info("SemanticCache: Daily 24-hour background flusher task initialized.")
    while True:
        try:
            await asyncio.sleep(86400)
            from core.cache.semantic_cache import clear_semantic_cache_chroma
            clear_semantic_cache_chroma()
            log.info("SemanticCache: Automated 24-hour ChromaDB semantic_cache flush completed.")
        except asyncio.CancelledError:
            log.info("SemanticCache: 24-hour background flusher task cancelled.")
            break
        except Exception as e:
            log.error(f"SemanticCache: Error during daily 24-hour cache flush: {e}")


from redis_store import redis_manager, redis_semantic_cache, RedisRateLimitMiddleware


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI application lifespan context manager.

    Args:
        app (FastAPI): Active FastAPI application instance.

    Yields:
        None: Controls application startup and shutdown lifecycle hooks.
    """
    try:
        await _start_mcp()
    except Exception as e:
        log.error(
            f"MCP startup failed ({e}). "
            "Requests will fall back to per-request spawn."
        )

    try:
        redis_client = await redis_manager.connect()
        if redis_client is not None:
            await redis_semantic_cache.init_index()
    except Exception as e:
        log.warning(f"Redis lifespan startup warning: {e}")

    flusher_task = asyncio.create_task(_daily_semantic_cache_flusher())

    yield

    flusher_task.cancel()
    try:
        await flusher_task
    except asyncio.CancelledError:
        pass

    await _stop_mcp()
    await redis_manager.close()


def create_app() -> FastAPI:
    """Build and configure the main FastAPI web application instance.

    Returns:
        FastAPI: Configured FastAPI application instance with routes and middleware registered.
    """
    from api.routes import router as query_router

    app = FastAPI(title="NL to SQL Chatbot API", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RedisRateLimitMiddleware)

    app.include_router(query_router)

    return app

