"""
main.py — FastAPI application entry point.

Starts the NL-to-SQL chatbot backend on port 8000.
All route logic lives in api/. All business logic lives in agents/, rag/, mcp/.

Run with:
    uvicorn main:app --reload --port 8000
"""


# ── MODULE TAG: FastAPI Server Entrypoint ──
import logging
from config.logger import setup_logging
from api.app_factory import create_app

# ── SYSTEM LOGGER INITIALIZATION ────────────────────────────────────────────────
# Configure root logger once at startup
setup_logging(level=logging.INFO)

# ── FASTAPI APPLICATION BOOTSTRAP ──────────────────────────────────────────────
# Build the FastAPI app (registers all routes: /status, /schema, /query)
app = create_app()

# ── CLI APPLICATION ENTRY POINT ─────────────────────────────────────────────────
# Trigger reload 2
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)