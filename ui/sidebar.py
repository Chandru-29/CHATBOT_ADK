"""
sidebar.py — Renders the Streamlit sidebar with connection status,
DB info, schema viewer, and utility buttons.

Call render_sidebar() from app.py after initialising session state.
"""


# ── MODULE TAG: Streamlit UI Sidebar Controls ──
import requests
import streamlit as st
from ui.session_init import (
    BACKEND_URL, OLLAMA_MODEL, DB_DIALECT, DB_HOST, DB_NAME,
    _fetch_schema, _check_backend_health,
)


def render_sidebar() -> None:
    """
    Draw the left sidebar containing:
      - App title and model info
      - Backend connection status check button
      - Connected database summary
      - Reload Schema button
      - Expandable schema viewer
      - Clear Chat button
    """
    with st.sidebar:
        # ── CAPTION & MAIN TITLE PANEL ────────────────────────────────────────────────
        st.markdown("###  CHATBOT")
        st.caption(f"Powered by Ollama `{OLLAMA_MODEL}` + MySQL")
        st.divider()

        # ── BACKEND SERVICE STATUS CHECKS ──────────────────────────────────────────────
        st.markdown("##### Service Status")
        if st.button("Check Connection", use_container_width=True):
            with st.spinner("Checking..."):
                ok, msg = _check_backend_health()
                st.session_state.ollama_ok  = ok
                st.session_state.ollama_msg = msg

        if st.session_state.ollama_msg:
            if st.session_state.ollama_ok:
                st.success("Backend connection active")
            else:
                st.error("Backend connection offline")

        st.divider()

        # ── DATABASE CONFIGURATION INFO & SCHEMA RELOADERS ─────────────────────────────
        st.markdown("##### Connected Database")
        st.info(
            f"**Dialect**: `{DB_DIALECT.upper()}`\n\n"
            f"**Host**: `{DB_HOST}`\n\n"
            f"**Database**: `{DB_NAME}`"
        )

        if st.button("🔄 Reload Schema", use_container_width=True, type="primary"):
            try:
                with st.spinner("Reloading schema..."):
                    schema = _fetch_schema()
                st.session_state.schema        = schema
                st.session_state.schema_loaded = True
                st.session_state.chat_history  = []
                st.success("Schema reloaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load schema: {str(e)}")

        # ── SCHEMA CATALOG VIEWER & CONVERSATION UTILITIES ─────────────────────────────
        if st.session_state.schema_loaded:
            st.divider()
            st.markdown("##### Schema Catalog")
            with st.expander("View full database schema", expanded=False):
                st.code(st.session_state.schema, language="sql")

            st.divider()
            if st.button("Clear Chat Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

