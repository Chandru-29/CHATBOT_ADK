"""
app.py — Streamlit frontend entry point.

Launches the chatbot UI. All rendering logic lives in src/ui/.
All API communication logic lives in src/ui/session_init.py and src/ui/chat_window.py.

Run with:
    streamlit run app.py
"""


# ── MODULE TAG: Streamlit UI Runner ──
import streamlit as st
from ui.page_styles import CSS
from ui.session_init import initialize_session_state
from ui.sidebar import render_sidebar
from ui.chat_window import render_chat, handle_input
from ui.right_drawer import render_right_drawer

# ── STREAMLIT PAGE CONFIGURATION ───────────────────────────────────────────────
st.set_page_config(
    page_title="CHATBOT",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── APPLICATION STYLES & STATE INITIALIZATION ──────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
initialize_session_state()

# ── FRONTEND COMPONENTS RENDERING & INPUT HANDLING ─────────────────────────────
render_sidebar()
render_right_drawer()  # Render drawer BEFORE query processing so toggle & drawer DOM exist during streaming
render_chat()
handle_input()
