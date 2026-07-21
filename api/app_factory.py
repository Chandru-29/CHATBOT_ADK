"""
app_factory.py — Creates the FastAPI application and registers all route modules.

Import create_app() from here in main.py to get the configured app instance.
Adding a new endpoint = create a new route file and add one include_router() here.

Change 2 — Persistent MCP Subprocess
--------------------------------------
A FastAPI `lifespan` context manager starts the MCP sql_executor subprocess
*once* at server startup and keeps it alive for the entire process lifetime.
This eliminates the ~200–500 ms per-request subprocess boot cost.

The singleton session is stored in `_mcp_state` and exposed via
`get_mcp_session()` / `restart_mcp()` so query_router.py can access it
without importing the full lifespan machinery.

`uvicorn --reload` behaviour: the lifespan runs on every reload so the
subprocess is automatically restarted. First post-reload request will
reconnect just as fast as a normal startup — identical to today's
per-request spawn.
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

from config.logger import get_logger
from database.engine import DB_URL

log = get_logger(__name__)

# ── Module-level MCP singleton ────────────────────────────────────────────────
# Populated by the lifespan; accessed via get_mcp_session() / restart_mcp().
_mcp_state: dict = {
    "read":    None,
    "write":   None,
    "session": None,
    "cm_stdio": None,    # stdio_client async context manager
    "cm_sess":  None,    # ClientSession async context manager
}

# ── SUBPROCESS INITIALIZATION UTILITIES ─────────────────────────────────────────

def build_server_params() -> StdioServerParameters:
    """
    Construct parameters to run the FastMCP sql_executor script as a python subprocess.
    """
    global _server_params
    # ── Get path to MCP executor script ──
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "mcp_server", "sql_executor.py"
    )

    # ── Return subprocess arguments ──
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
    log.info("MCP persistent session initialized (sql_executor subprocess running)")


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


# ── Public helpers (used by query_router.py) ───────────────────────────────────

def get_mcp_session() -> ClientSession | None:
    """Return the active persistent MCP ClientSession, or None if not ready."""
    # ── Return active MCP session ──
    return _mcp_state.get("session")


async def restart_mcp() -> ClientSession:
    """
    Teardown and restart the MCP subprocess.
    Called by query_router.py when a session.call_tool() raises an exception.
    Returns the fresh session.
    """
    log.warning("MCP restart triggered — reconnecting to sql_executor subprocess…")
    # ── Restart persistent MCP session ──
    await _stop_mcp()
    await _start_mcp()
    return _mcp_state["session"]


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start persistent MCP on startup; cleanly close on shutdown."""
    try:
        await _start_mcp()
    except Exception as e:
        log.error(
            f"MCP startup failed ({e}). "
            "Requests will fall back to per-request spawn."
        )
    yield
    await _stop_mcp()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    All routes are registered here. CORS is set to allow all origins so
    the Streamlit frontend can call the API from any port.
    """
    from api.query_router  import router as query_router

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
