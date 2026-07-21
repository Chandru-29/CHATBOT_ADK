"""
chat_window.py — Renders the chat history and handles new user input.

render_chat()  — Draws all previous messages with agent badges.
handle_input() — Reads new user message, calls backend, appends reply.

Call both from app.py after render_sidebar().

Change 7 — Streaming output (opt-in)
--------------------------------------
_send_query() now has a `stream` parameter. When stream=True it uses
httpx.Client().stream() + st.write_stream() for token-by-token rendering.
The default (stream=False) preserves the existing requests.post() behaviour
so nothing breaks.

DEPENDENCY: `pip install httpx` (usually already present via FastAPI/Starlette).
"""

# ── MODULE TAG: Streamlit UI Chat Window Rendering ──
import requests
import streamlit as st
from ui.session_init import BACKEND_URL, DB_NAME, OLLAMA_MODEL

# Agent badge styles: maps agent_name → (background, text_color, icon)
AGENT_STYLES: dict[str, tuple] = {
    "HR Specialist":            ("#e6f4ea", "#137333", "💼"),
    "Sales Specialist":         ("#f3e8fd", "#681da8", "🛒"),
    "Cross-Domain Coordinator": ("#fef7e0", "#b06000", "🧩"),
    "General Agent":            ("#f1f3f4", "#5f6368", "💬"),
    "System Agent":             ("#fce8e6", "#c5221f", "⚠️"),
}

# ── Streaming opt-in toggle ───────────────────────────────────────────────────
# Set to True to enable token-by-token streaming via httpx.
# Requires: pip install httpx
_USE_STREAMING = True  # flip to True to activate Change 7


# ── BACKEND WEB SERVICE REQUEST INTERFACES ─────────────────────────────────────
def _send_query(user_question: str, schema: str, chat_history: list) -> dict:
    """
    POST the user's question to the backend /query endpoint.
    Returns a result dict; returns an error dict on any network failure.

    When _USE_STREAMING is True, delegates to _send_query_streaming() which
    renders tokens progressively inside the current st.chat_message context.
    """
    try:
        payload = {
            "user_question": user_question,
            "db_schema":     schema,
            "chat_history":  chat_history,
        }
        response = requests.post(f"{BACKEND_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": response.json().get("detail", f"Backend error {response.status_code}"),
            "attempts": 0, "agent_name": "System Agent",
        }
    except Exception as e:
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": f"Failed to connect to backend: {e}",
            "attempts": 0, "agent_name": "System Agent",
        }


def _send_query_streaming(user_question: str, schema: str, chat_history: list, badge_placeholder=None) -> dict:
    """
    Change 7 — Stream tokens from /query?stream=true using httpx.

    Renders tokens progressively via st.write_stream() inside the current
    st.chat_message("assistant") context.

    Returns a dictionary of:
        {"content": ..., "agent_name": ..., "sql": ...}
    Falls back to blocking request on any httpx import failure.
    """
    import time
    t_start = time.perf_counter()

    try:
        import httpx
    except ImportError:
        # httpx not installed — fall back to blocking request silently
        result = _send_query(user_question, schema, chat_history)
        block_latency = int((time.perf_counter() - t_start) * 1000)
        return {
            "content": result.get("natural_answer") or result.get("error") or "",
            "agent_name": result.get("agent_name", "General Agent"),
            "sql": result.get("sql"),
            "latency_ms": block_latency
        }

    payload = {
        "user_question": user_question,
        "db_schema":     schema,
        "chat_history":  chat_history,
    }
    collected_tokens: list[str] = []
    info = {"agent_name": "General Agent", "sql": None, "latency_ms": None}

    def _token_generator():
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{BACKEND_URL}/query?stream=true",
                json=payload,
            ) as response:
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    response.read()
                    try:
                        data = response.json()
                        info["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
                        ans = data.get("natural_answer") or data.get("error") or "I'm not sure how to answer that."
                        info["agent_name"] = data.get("agent_name", "General Agent")
                        info["sql"] = data.get("sql")
                        collected_tokens.append(ans)
                        yield ans
                    except Exception as e:
                        err_msg = f"Failed to parse JSON response: {e}"
                        collected_tokens.append(err_msg)
                        yield err_msg
                else:
                    agent_header = response.headers.get("x-agent-name")
                    sql_header = response.headers.get("x-sql-query")
                    if agent_header:
                        info["agent_name"] = agent_header
                        if badge_placeholder:
                            badge_placeholder.markdown(_agent_badge(agent_header, is_loading=True), unsafe_allow_html=True)
                    if sql_header:
                        info["sql"] = sql_header

                    first_chunk = True
                    for chunk in response.iter_text():
                        if chunk:
                            if first_chunk:
                                info["latency_ms"] = int((time.perf_counter() - t_start) * 1000)
                                first_chunk = False
                            collected_tokens.append(chunk)
                            yield chunk

    try:
        st.write_stream(_token_generator())
    except Exception as e:
        err_msg = f"⚠️ Connection to backend lost or failed: {e}"
        st.error(err_msg)
        collected_tokens.append(err_msg)

    if info.get("latency_ms") is None:
        info["latency_ms"] = int((time.perf_counter() - t_start) * 1000)

    return {
        "content": "".join(collected_tokens),
        "agent_name": info["agent_name"],
        "sql": info["sql"],
        "latency_ms": info["latency_ms"]
    }


# ── AGENT LATENCY BADGES RENDERING ENGINE ───────────────────────────────────────
def _agent_badge(agent_name: str, latency_ms: int = None, is_loading: bool = False) -> str:
    """Return the HTML string for the coloured agent badge and optional latency info/loading animation."""
    bg, text_color, icon = AGENT_STYLES.get(agent_name, ("#f1f3f4", "#5f6368", "🤖"))
    badge_html = (
        f"<span style='font-size: 11px; font-weight: 500; "
        f"background-color: {bg}; color: {text_color}; "
        f"padding: 2px 8px; border-radius: 12px; "
        f"border: 1px solid {text_color}33; "
        f"margin-bottom: 6px; display: inline-flex; "
        f"align-items: center; gap: 4px;'>"
        f"<span>{icon}</span> <span>{agent_name}</span></span>"
    )
    if is_loading:
        badge_html += (
            f"<style>"
            f"@keyframes badge-bounce {{"
            f"  0%, 80%, 100% {{ transform: scale(0); }}"
            f"  40% {{ transform: scale(1.0); }}"
            f"}}"
            f".badge-loading {{"
            f"  display: inline-flex;"
            f"  align-items: center;"
            f"  gap: 2px;"
            f"  margin-left: 6px;"
            f"  vertical-align: middle;"
            f"}}"
            f".badge-loading span {{"
            f"  width: 4px;"
            f"  height: 4px;"
            f"  background-color: {text_color};"
            f"  border-radius: 50%;"
            f"  display: inline-block;"
            f"  animation: badge-bounce 1.4s infinite ease-in-out both;"
            f"}}"
            f".badge-loading span:nth-child(2) {{ animation-delay: 0.2s; }}"
            f".badge-loading span:nth-child(3) {{ animation-delay: 0.4s; }}"
            f"</style>"
            f"<span class='badge-loading'><span></span><span></span><span></span></span>"
        )
    elif latency_ms is not None:
        if latency_ms >= 1000:
            latency_str = f"{latency_ms / 1000:.2f}s"
        else:
            latency_str = f"{latency_ms}ms"
        badge_html += (
            f" <span style='font-size: 11px; color: #71717a; "
            f"font-family: monospace; margin-left: 6px; "
            f"vertical-align: middle; display: inline-flex; "
            f"align-items: center;'>"
            f"⏱ {latency_str}</span>"
        )
    return badge_html


# ── STREAMLIT CHAT THREAD VIEW RENDERER ──────────────────────────────────────────
def render_chat() -> None:
    """
    Draw all messages in st.session_state['chat_history'].
    Shows the page title and status pills when the schema is loaded.
    Shows a 'connection failed' placeholder when it isn't.
    """
    st.markdown(
        "<h2 style='text-align: center; font-weight: 600; margin-bottom: 2px;'>"
        " CHATBOT</h2>",
        unsafe_allow_html=True,
    )

    if not st.session_state.schema_loaded:
        st.markdown("""
        <div style='text-align:center; padding:100px 20px;'>
            <h3 style='color:#18181b; font-weight:600; font-size:1.5rem; margin-bottom:8px;'>
                Database Connection Failed</h3>
            <p style='color:#71717a; font-size:0.95rem; margin:0;'>
                Could not load the database schema. Please check if your MySQL
                server is running and configuration in .env is correct, then
                click <b>Reload Schema</b> in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Status pills row
    tables_count = 0
    try:
        tables_count = len([
            line for line in st.session_state.schema.split("\n")
            if line.startswith("Table:")
        ])
    except Exception:
        pass

    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 24px;'>
        <span class='status-pill'><span style='color:#10a37f;'>●</span> DB: <b>{DB_NAME}</b></span>
        <span class='status-pill'><span style='color:#a855f7;'>●</span> Engine: <b>{OLLAMA_MODEL}</b></span>
        <span class='status-pill'><span style='color:#3b82f6;'>●</span> Schema: <b>{tables_count} Tables</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            if role == "assistant" and msg.get("agent_name"):
                st.markdown(_agent_badge(msg["agent_name"], msg.get("latency_ms")), unsafe_allow_html=True)
            st.markdown(msg["content"])


# ── FRONTEND CHAT FORM EVENT DISPATCHER ─────────────────────────────────────────
def handle_input() -> None:
    """
    Render the chat input box and process new messages.
    Shows a spinner while waiting for the backend, then appends the reply.

    When _USE_STREAMING is True (Change 7), renders tokens progressively
    via st.write_stream() instead of showing a spinner.
    """
    if not st.session_state.schema_loaded:
        return

    is_generating = st.session_state.get("generating", False)
    user_input = st.chat_input("Ask a question about your database...", disabled=is_generating)

    if user_input:
        st.session_state["pending_input"] = user_input
        st.session_state["generating"] = True
        st.rerun()

    if is_generating and st.session_state.get("pending_input"):
        pending_input = st.session_state["pending_input"]

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(pending_input)

        st.session_state.chat_history.append({"role": "user", "content": pending_input})

        history_to_send = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]

        import time
        t_start = time.perf_counter()

        try:
            with st.chat_message("assistant"):

                if _USE_STREAMING:
                    # ── Change 7: streaming path ──────────────────────────────────
                    # st.write_stream() renders tokens as they arrive.
                    # We still need an agent name; default to a placeholder.
                    agent_name = "General Agent"
                    badge_placeholder = st.empty()
                    badge_placeholder.markdown(_agent_badge(agent_name, is_loading=True), unsafe_allow_html=True)
                    
                    result = _send_query_streaming(pending_input, st.session_state.schema, history_to_send, badge_placeholder)
                    content = result["content"]
                    agent_name = result["agent_name"]
                    
                    if content.startswith("⚠️"):
                        agent_name = "System Agent"
                        
                    latency_ms = result.get("latency_ms") or int((time.perf_counter() - t_start) * 1000)
                    badge_placeholder.markdown(_agent_badge(agent_name, latency_ms), unsafe_allow_html=True)
                else:
                    # ── Blocking path (default) ───────────────────────────────────
                    with st.spinner("Thinking..."):
                        result = _send_query(
                            user_question=pending_input,
                            schema=st.session_state.schema,
                            chat_history=history_to_send,
                        )

                    content    = result.get("natural_answer") or result.get("error") or "I'm not sure how to answer that."
                    agent_name = result.get("agent_name", "General Agent")
                    if result.get("error"):
                        content    = f"⚠️ {content}"
                        agent_name = "System Agent"

                    latency_ms = int((time.perf_counter() - t_start) * 1000)
                    st.markdown(_agent_badge(agent_name, latency_ms), unsafe_allow_html=True)
                    st.markdown(content)

            st.session_state.chat_history.append({
                "role":       "assistant",
                "content":    content,
                "agent_name": agent_name,
                "latency_ms": latency_ms,
            })
        finally:
            st.session_state["generating"] = False
            st.session_state["pending_input"] = None

        st.rerun()
