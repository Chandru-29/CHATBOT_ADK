"""
engine.py — Database engine stub.

The WMS chatbot uses an HTTP API for all database operations (WMS_API_URL in .env).
SQLAlchemy is no longer used for query execution or schema reads.
"""

from core.config.settings import WMS_API_URL
from core.config.logger import get_logger

log = get_logger(__name__)


class _DummyEngine:
    """Lightweight stub that mimics a SQLAlchemy Engine interface."""

    def connect(self):
        raise RuntimeError(
            "Direct DB connections are disabled. "
            f"All queries must go through the WMS HTTP API: {WMS_API_URL}"
        )

    def dispose(self):
        pass


engine = _DummyEngine()
DB_URL = WMS_API_URL

log.info("Database engine: HTTP API mode (WMS execute-query endpoint)")
