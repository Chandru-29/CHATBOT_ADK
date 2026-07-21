"""
session_init.py — Sets up all st.session_state keys on the first app load.

Called once at the top of app.py before rendering any UI components.
Separating this from the rendering code makes it easy to see exactly
what state the app depends on.
"""


# ── MODULE TAG: Streamlit Session State Initialization ──
import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL  = "http://localhost:8000"
OLLAMA_MODEL = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")

DB_DIALECT = os.getenv("DB_DIALECT", "mysql")
DB_HOST    = os.getenv("DB_HOST",    "localhost")
DB_NAME    = os.getenv("DB_NAME",    "company_data")


# ── BACKEND API INTEGRATIONS ────────────────────────────────────────────────────

def _fetch_schema() -> str:
    """Fetch the DB schema from the backend API. Raises RuntimeError on failure."""
    response = requests.get(f"{BACKEND_URL}/schema")
    if response.status_code == 200:
        return response.json()["schema"]
    raise RuntimeError(response.json().get("detail", "Failed to retrieve schema"))


def _check_backend_health() -> tuple[bool, str]:
    """Ping the backend API. Returns (is_ok, message)."""
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            data = response.json()
            return data["ok"], data["message"]
        return False, f"Backend returned status code {response.status_code}"
    except Exception as e:
        return False, f"Cannot connect to backend: {e}"


# ── STREAMLIT STATE INITIALIZER ─────────────────────────────────────────────────
def initialize_session_state() -> None:

    """
    Populate all required st.session_state keys on the very first run.
    Skips silently if already initialised so Streamlit re-runs don't reset state.
    """
    if "chat_history" in st.session_state:
        return

    default_schema = ""
    default_loaded = False
    try:
        default_schema = _fetch_schema()
        default_loaded = True
    except Exception:
        pass

    try:
        ollama_ok, ollama_msg = _check_backend_health()
    except Exception as e:
        ollama_ok, ollama_msg = False, str(e)

    st.session_state["chat_history"]   = []
    st.session_state["schema"]         = default_schema
    st.session_state["ollama_ok"]      = ollama_ok
    st.session_state["ollama_msg"]     = ollama_msg
    st.session_state["schema_loaded"]  = default_loaded
    st.session_state["generating"]     = False
    st.session_state["pending_input"]  = None
