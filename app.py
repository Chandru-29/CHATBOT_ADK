import streamlit as st
import pandas as pd
import time
from pathlib import Path
import requests

BACKEND_URL = "http://localhost:8000"
OLLAMA_MODEL = "llama3.1"
MAX_ROWS = 500

def check_ollama_status() -> tuple[bool, str]:
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        if response.status_code == 200:
            data = response.json()
            return data["ok"], data["message"]
        return False, f"Backend returned status code {response.status_code}"
    except Exception as e:
        return False, f"Cannot connect to backend: {e}"

def get_schema() -> str:
    try:
        response = requests.get(f"{BACKEND_URL}/schema")
        if response.status_code == 200:
            return response.json()["schema"]
        raise RuntimeError(response.json().get("detail", "Failed to retrieve schema"))
    except Exception as e:
        raise RuntimeError(f"Error fetching schema from backend: {e}")

def process_query(user_question: str, schema: str, chat_history: list = None) -> dict:
    try:
        payload = {
            "user_question": user_question,
            "db_schema": schema,
            "chat_history": chat_history or []
        }
        response = requests.post(f"{BACKEND_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": response.json().get("detail", f"Backend error {response.status_code}"),
            "attempts": 0
        }
    except Exception as e:
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": f"Failed to connect to backend: {e}",
            "attempts": 0
        }

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CHATBOT",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (ChatGPT & Claude Minimalist Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply clean typography globally */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f9f9fb !important; /* Clean off-white */
    color: #18181b !important;
}

/* Force headings and text to be dark grey/black in main content */
h1, h2, h3, h4, h5, h6, p, span, li {
    color: #18181b !important;
}

/* Fix top header white bar */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    background-color: #f9f9fb !important;
    background: #f9f9fb !important;
}

/* Fix bottom chat input container white bar */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background-color: #f9f9fb !important;
    background: #f9f9fb !important;
}

/* Align user chat message container to the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

/* Fix avatar margins when row is reversed for user */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
    margin-left: 12px !important;
    margin-right: 0 !important;
}

/* Remove default message borders and backgrounds */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* User chat bubble styling - aligned right, bottom-right sharp */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background-color: #f4f4f5 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 18px !important;
    border: 1px solid #e4e4e7 !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    color: #18181b !important;
    width: fit-content !important;
    max-width: 75% !important;
    flex-grow: 0 !important;
}

/* Bot chat bubble styling - aligned left, bottom-left sharp */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background-color: #ffffff !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 14px 20px !important;
    border: 1px solid #e4e4e7 !important;
    color: #18181b !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    width: fit-content !important;
    max-width: 75% !important;
    flex-grow: 0 !important;
}

/* Sidebar styling - minimalist neutral */
[data-testid="stSidebar"] {
    border-right: 1px solid #e4e4e7 !important;
    background-color: #f3f3f5 !important; /* Cool grey sidebar */
}

/* Sidebar elements text and headings override */
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h5, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label {
    color: #18181b !important;
}

/* Ensure text inputs and buttons in sidebar have good borders/colors */
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #d4d4d8 !important;
    border-radius: 6px !important;
}

/* Custom width for content column */
.block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 7rem !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #e4e4e7;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #d4d4d8;
}

/* Clean buttons - flat styling */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    border: 1px solid #e4e4e7 !important;
    background: #ffffff !important;
    color: #18181b !important;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: #f4f4f5 !important;
    border-color: #d4d4d8 !important;
}
.stButton > button[kind="primary"] {
    background: #10a37f !important; /* ChatGPT Brand Green */
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1a7f64 !important;
}

/* Minimal Table styling for markdown outputs */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e4e4e7;
}
th {
    background: #f4f4f5;
    color: #18181b;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1.5px solid #e4e4e7;
}
td {
    padding: 8px 14px;
    border-bottom: 1px solid #f4f4f5;
    color: #27272a;
}
tr:hover {
    background-color: #fafafa;
}

/* Hide standard Streamlit decorations */
footer { visibility: hidden; }
[data-testid="stElementToolbar"] { display: none !important; }

/* Status pill row styling */
.status-pill {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #71717a;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
</style>

""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

DB_DIALECT = os.getenv("DB_DIALECT", "mysql")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "company_data")

def init_state():
    if "chat_history" in st.session_state:
        return
        
    default_schema = ""
    default_loaded = False
    try:
        default_schema = get_schema()
        default_loaded = True
    except Exception:
        pass

    try:
        ollama_ok, ollama_msg = check_ollama_status()
    except Exception as e:
        ollama_ok, ollama_msg = False, str(e)

    st.session_state["chat_history"] = []
    st.session_state["schema"] = default_schema
    st.session_state["ollama_ok"] = ollama_ok
    st.session_state["ollama_msg"] = ollama_msg
    st.session_state["schema_loaded"] = default_loaded

init_state()

# ─────────────────────────────────────────────
# SIDEBAR (Minimalist Settings)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐦 CHATBOT")
    st.caption(f"Powered by Ollama `{OLLAMA_MODEL}` + MySQL")
    st.divider()

    # Connection status
    st.markdown("##### Service Status")
    if st.button("Check Connection", use_container_width=True):
        with st.spinner("Checking..."):
            ok, msg = check_ollama_status()
            st.session_state.ollama_ok  = ok
            st.session_state.ollama_msg = msg

    if st.session_state.ollama_msg:
        if st.session_state.ollama_ok:
            st.success("Backend connection active")
        else:
            st.error("Backend connection offline")

    st.divider()

    # Database Configuration Summary
    st.markdown("##### Connected Database")
    st.info(f"**Dialect**: `{DB_DIALECT.upper()}`\n\n**Host**: `{DB_HOST}`\n\n**Database**: `{DB_NAME}`")
    
    if st.button("🔄 Reload Schema", use_container_width=True, type="primary"):
        try:
            with st.spinner("Reloading schema..."):
                schema = get_schema()
            st.session_state.schema        = schema
            st.session_state.schema_loaded = True
            st.session_state.chat_history  = []
            st.success("Schema reloaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load schema: {str(e)}")

    # Schema viewer
    if st.session_state.schema_loaded:
        st.divider()
        st.markdown("##### Schema Catalog")
        with st.expander("View full database schema", expanded=False):
            st.code(st.session_state.schema, language="sql")

        st.divider()
        if st.button("Clear Chat Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
# Minimal Title
st.markdown("<h2 style='text-align: center; font-weight: 600; margin-bottom: 2px;'>🐦 CHATBOT</h2>", unsafe_allow_html=True)

if not st.session_state.schema_loaded:
    # Minimalist Welcome screen
    st.markdown("""
    <div style='text-align:center; padding:100px 20px;'>
        <h3 style='color:#18181b; font-weight:600; font-size:1.5rem; margin-bottom:8px;'>Database Connection Failed</h3>
        <p style='color:#71717a; font-size:0.95rem; margin:0;'>Could not load the database schema. Please check if your MySQL server is running and configuration in .env is correct, then click <b>Reload Schema</b> in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Count tables in the schema
    tables_count = 0
    try:
        tables_count = len([line for line in st.session_state.schema.split("\n") if line.startswith("Table:")])
    except Exception:
        pass

    # Status Pills Row (Claude style status bar)
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 24px;'>
        <span class='status-pill'><span style='color:#10a37f;'>●</span> DB: <b>{DB_NAME}</b></span>
        <span class='status-pill'><span style='color:#a855f7;'>●</span> Engine: <b>{OLLAMA_MODEL}</b></span>
        <span class='status-pill'><span style='color:#3b82f6;'>●</span> Schema: <b>{tables_count} Tables</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Render Chat History ──
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # ── Bottom Floating Input (ChatGPT Style) ──
    user_input = st.chat_input("Ask a question about your database...")

    if user_input:
        # 1. Render user message instantly
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # 2. Get backend response with inline status spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_to_send = []
                for m in st.session_state.chat_history[:-1]:
                    history_to_send.append({"role": m["role"], "content": m["content"]})
                
                result = process_query(
                    user_question=user_input,
                    schema=st.session_state.schema,
                    chat_history=history_to_send
                )

                content = result.get("natural_answer") or result.get("error") or "I'm not sure how to answer that."
                if result.get("error"):
                    content = f"⚠️ {content}"

                st.markdown(content)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": content
                })
        st.rerun()
