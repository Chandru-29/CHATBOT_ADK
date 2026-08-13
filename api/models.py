"""
models.py — Pydantic request/response models for the FastAPI endpoints.

Centralised here so all routes import from one place instead of
each defining their own inline models.
"""

# ── MODULE TAG: API Request Schemas ──
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Incoming request payload model for the POST /query endpoint.

    Attributes:
        user_question (str): User natural language question string.
        db_schema (str, optional): Database schema string. Defaults to "".
        chat_history (list, optional): Previous chat turns list. Defaults to [].
        query (str, optional): Alias query string. Defaults to "".
        session_id (str, optional): Session identifier string. Defaults to "default".
    """
    user_question: str
    db_schema:     str  = ""
    chat_history:  list = []
    query: str = ""
    session_id:    str  = "default"

