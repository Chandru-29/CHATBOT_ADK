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


def build_server_params() -> StdioServerParameters:
    """Construct parameters to run FastMCP server script as a python subprocess."""
    server_script = os.path.join(
        PROJECT_ROOT, "mcp_service", "server.py"
    )

    return StdioServerParameters(
        command=sys.executable,
        args=[server_script, DB_URL],
    )


async def _start_mcp() -> None:
    """Spawn the MCP subprocess and initialise a persistent ClientSession."""
    global _server_params
    _server_params = build_server_params()

    cm_stdio = stdio_client(_server_params)
    read, write = await cm_stdio.__aenter__()

    cm_sess = ClientSession(read, write)
    session = await cm_sess.__aenter__()

    await session.initialize()
    await session.list_tools()

    _mcp_state["read"]     = read
    _mcp_state["write"]    = write
    _mcp_state["session"]  = session
    _mcp_state["cm_stdio"] = cm_stdio
    _mcp_state["cm_sess"]  = cm_sess
    log.info("MCP persistent session initialized (src/domain/mcp/server.py subprocess running)")


async def _stop_mcp() -> None:
    """Cleanly shut down the persistent MCP session and subprocess."""
    try:
        if _mcp_state["cm_sess"]:
            await _mcp_state["cm_sess"].__aexit__(None, None, None)
    except Exception as e:
        log.warning(f"MCP session close warning: {e}")
    try:
        if _mcp_state["cm_stdio"]:
            await _mcp_state["cm_stdio"].__aexit__(None, None, None)
    except Exception as e:
        log.warning(f"MCP stdio close warning: {e}")
    for key in ("read", "write", "session", "cm_stdio", "cm_sess"):
        _mcp_state[key] = None
    log.info("MCP persistent session closed")


def get_mcp_session() -> ClientSession | None:
    """Return the active persistent MCP ClientSession, or None if not ready."""
    return _mcp_state.get("session")


async def restart_mcp() -> ClientSession:
    """Teardown and restart the MCP subprocess."""
    log.warning("MCP restart triggered — reconnecting to src/domain/mcp/server.py subprocess…")
    await _stop_mcp()
    await _start_mcp()
    return _mcp_state["session"]


async def _daily_semantic_cache_flusher() -> None:
    """Background worker task that automatically flushes the ChromaDB 'semantic_cache' collection every 24 hours."""
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start persistent MCP & background 24h cache flusher on startup; cleanly close on shutdown."""
    try:
        await _start_mcp()
    except Exception as e:
        log.error(
            f"MCP startup failed ({e}). "
            "Requests will fall back to per-request spawn."
        )

    flusher_task = asyncio.create_task(_daily_semantic_cache_flusher())

    yield

    flusher_task.cancel()
    try:
        await flusher_task
    except asyncio.CancelledError:
        pass

    await _stop_mcp()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    from api.routes import router as query_router

    app = FastAPI(title="NL to SQL Chatbot API", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router)

    return app
