"""
models.py — Pydantic request/response models for the FastAPI endpoints.

Centralised here so all routes import from one place instead of
each defining their own inline models.
"""

# ── MODULE TAG: API Request Schemas ──
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Incoming payload for the POST /query endpoint."""
    user_question: str
    db_schema:     str  = ""
    chat_history:  list = []
    query: str = ""
