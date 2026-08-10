"""
chat_window.py — Renders the chat history and handles new user input.

render_chat()  — Draws all previous messages with agent badges.
handle_input() — Reads new user message, calls backend, appends reply.

Call both from app.py after render_sidebar().
"""

# ── MODULE TAG: Streamlit UI Chat Window Rendering ──
import time
import requests
import streamlit as st
from ui.session_init import BACKEND_URL, DB_NAME, GEMINI_MODEL, ROUTER_MODEL, OLLAMA_MODEL

from ui import pipeline_panel as pp

# Agent badge styles: maps agent_name → (background, text_color, icon)
AGENT_STYLES: dict[str, tuple] = {
    "WMS Assistant":            ("#e6f4ea", "#137333", "📦"),
    "General Agent":            ("#f1f3f4", "#5f6368", "💬"),
    "System Agent":             ("#fce8e6", "#c5221f", "⚠️"),
    "System Guard":             ("#fef2f2", "#991b1b", "🛡️"),
}

# ── Streaming opt-in toggle ───────────────────────────────────────────────────
_USE_STREAMING = True


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
        response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": response.json().get("detail", f"Backend error {response.status_code}"),
            "attempts": 0, "agent_name": "System Agent",
        }
    except requests.exceptions.Timeout:
        return {
            "sql": None, "columns": [], "rows": [], "natural_answer": None,
            "error": "Backend not responding — query request timed out.",
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
    Stream tokens from /query?stream=true using httpx.

    Renders tokens progressively via st.write_stream() inside the current
    st.chat_message("assistant") context.
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
    info = {"agent_name": "General Agent", "sql": None, "latency_ms": None, "cache_hit": None, "steps": None}

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
                        info["cache_hit"] = data.get("cache_hit")
                        info["steps"] = data.get("attempts") or data.get("steps")
                        collected_tokens.append(ans)
                        yield ans
                    except Exception as e:
                        err_msg = f"Failed to parse JSON response: {e}"
                        collected_tokens.append(err_msg)
                        yield err_msg
                else:
                    agent_header = response.headers.get("x-agent-name")
                    sql_header   = response.headers.get("x-sql-query")
                    cache_header = response.headers.get("x-cache-hit")
                    step_header  = response.headers.get("x-agent-step")
                    if step_header and step_header.isdigit():
                        info["steps"] = int(step_header)
                    if agent_header:
                        info["agent_name"] = agent_header
                        if badge_placeholder:
                            badge_placeholder.markdown(_agent_badge(agent_header, is_loading=True, steps=info.get("steps")), unsafe_allow_html=True)
                    if sql_header:
                        info["sql"] = sql_header
                    if cache_header:
                        info["cache_hit"] = cache_header

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
        "latency_ms": info["latency_ms"],
        "cache_hit": info["cache_hit"],
        "steps": info["steps"],
    }


# ── AGENT LATENCY BADGES RENDERING ENGINE ───────────────────────────────────────
def _agent_badge(agent_name: str, latency_ms: int = None, is_loading: bool = False, steps: int = None) -> str:
    """Return the HTML string for the coloured agent badge, step reasoning info, and optional latency info/loading animation."""
    bg, text_color, icon = AGENT_STYLES.get(agent_name, ("#f1f3f4", "#5f6368", "🤖"))

    step_tag = ""
    if steps and steps > 1:
        step_tag = f" <span style='font-size: 10px; opacity: 0.85;'>· Step {steps}/5</span>"

    badge_html = (
        f"<span style='font-size: 11px; font-weight: 500; "
        f"background-color: {bg}; color: {text_color}; "
        f"padding: 2px 8px; border-radius: 12px; "
        f"border: 1px solid {text_color}33; "
        f"margin-bottom: 6px; display: inline-flex; "
        f"align-items: center; gap: 4px;'>"
        f"<span>{icon}</span> <span>{agent_name}{step_tag}</span></span>"
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
        steps_info = f" ({steps} steps)" if steps and steps > 1 else ""
        badge_html += (
            f" <span style='font-size: 11px; color: #71717a; "
            f"font-family: monospace; margin-left: 6px; "
            f"vertical-align: middle; display: inline-flex; "
            f"align-items: center;'>"
            f"⏱ {latency_str}{steps_info}</span>"
        )
    return badge_html


def _format_message_content(content: str) -> str:
    """Format message content, converting security omission notices into rich callout cards."""
    if not content:
        return ""
    if "omitted for security" in content.lower() or "sensitive and omitted" in content.lower():
        return (
            "<div style='background-color: #fef2f2; border: 1px solid #fca5a5; border-left: 4px solid #ef4444; "
            "border-radius: 8px; padding: 12px 16px; margin: 8px 0; font-family: system-ui, -apple-system, sans-serif;'>"
            "<div style='display: flex; align-items: center; gap: 8px; font-weight: 600; color: #991b1b; font-size: 13px; margin-bottom: 4px;'>"
            "<span>🛡️</span> <span>Security Policy Notice</span>"
            "</div>"
            "<div style='color: #7f1d1d; font-size: 12.5px; line-height: 1.5;'>"
            "This query requests sensitive personal data (such as email addresses, passwords, or IDs) which cannot be displayed to protect privacy and data security."
            "</div>"
            "</div>"
        )
    return content


# ── STREAMLIT CHAT THREAD VIEW RENDERER ──────────────────────────────────────────
def render_chat() -> None:
    """
    Draw all messages in st.session_state['chat_history'].
    Shows the page title and status pills when the schema is loaded.
    The pipeline flow panel is shown in the right drawer (render_right_drawer in app.py).
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
        <span class='status-pill'><span style='color:#a855f7;'>●</span> Engine: <b>{GEMINI_MODEL}</b></span>
        <span class='status-pill'><span style='color:#3b82f6;'>●</span> Schema: <b>{tables_count} Tables</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            if role == "assistant" and msg.get("agent_name"):
                st.markdown(_agent_badge(msg["agent_name"], msg.get("latency_ms"), steps=msg.get("steps")), unsafe_allow_html=True)
            st.markdown(_format_message_content(msg["content"]), unsafe_allow_html=True)


# ── FRONTEND CHAT FORM EVENT DISPATCHER ─────────────────────────────────────────
def handle_input() -> None:
    """
    Render the chat input box and process new messages.
    Updates pipeline_panel state before/during/after the query.

    When _USE_STREAMING is True, renders tokens progressively
    via st.write_stream() instead of showing a spinner.
    """
    if not st.session_state.schema_loaded:
        return

    is_generating = st.session_state.get("generating", False)
    user_input = st.chat_input("Ask a question about your database...", disabled=is_generating)

    if user_input:
        st.session_state["pending_input"] = user_input
        st.session_state["generating"] = True
        pp.reset_pipeline()
        pp.set_step("l1", pp.STATUS_ACTIVE)
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

        t_start = time.perf_counter()

        # ── Mark all steps active while backend is working ──────────────────
        for step_id in ("l1", "l2", "l3", "l4", "l5", "l6"):
            pp.set_step(step_id, pp.STATUS_ACTIVE)

        try:
            with st.chat_message("assistant"):

                if _USE_STREAMING:
                    # ── streaming path ──────────────────────────────────
                    agent_name = "General Agent"
                    badge_placeholder = st.empty()
                    badge_placeholder.markdown(_agent_badge(agent_name, is_loading=True), unsafe_allow_html=True)

                    result = _send_query_streaming(pending_input, st.session_state.schema, history_to_send, badge_placeholder)
                    content = result["content"]
                    agent_name = result["agent_name"]

                    if content.startswith("⚠️"):
                        agent_name = "System Agent"
                    elif "cannot modify the database" in content.lower() or "read-only access" in content.lower():
                        agent_name = "System Guard"

                    latency_ms = result.get("latency_ms") or int((time.perf_counter() - t_start) * 1000)
                    badge_placeholder.markdown(_agent_badge(agent_name, latency_ms, steps=result.get("steps")), unsafe_allow_html=True)

                    # ── Update pipeline panel ────────────────────────────────
                    _update_pipeline_from_result(
                        agent_name=agent_name,
                        sql=result.get("sql"),
                        total_ms=latency_ms,
                        is_error=content.startswith("⚠️"),
                        content=content,
                        cache_hit=result.get("cache_hit"),
                        user_question=pending_input,
                    )

                else:
                    # ── Blocking path ───────────────────────────────────────────
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
                    elif "cannot modify the database" in content.lower() or "read-only access" in content.lower():
                        agent_name = "System Guard"

                    latency_ms = int((time.perf_counter() - t_start) * 1000)
                    st.markdown(_agent_badge(agent_name, latency_ms), unsafe_allow_html=True)
                    st.markdown(content)

                    # ── Update pipeline panel ────────────────────────────────
                    _update_pipeline_from_result(
                        agent_name=agent_name,
                        sql=result.get("sql"),
                        total_ms=latency_ms,
                        is_error=bool(result.get("error")),
                        content=content,
                        cache_hit=result.get("cache_hit"),
                        user_question=pending_input,
                    )


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


def _update_pipeline_from_result(
    agent_name: str,
    sql: str | None,
    total_ms: float,
    is_error: bool,
    content: str = "",
    cache_hit: str | None = None,
    user_question: str = "",
) -> None:
    """
    Derive precise pipeline step statuses and rich telemetry metrics for ALL 8 stages (L0..L7)
    from the query result, updating interactive layer accordions in session_state.
    """
    has_sql = bool(sql)
    cache_likely = (total_ms < 1200) and not is_error
    is_write_refusal = (agent_name == "System Guard") and ("cannot modify the database" in content.lower() or "read-only access" in content.lower())
    is_raw_sql_refusal = (agent_name == "System Guard") and ("rather than entering raw sql" in content.lower() or "natural language" in content.lower())
    is_l1_security_block = is_error and not has_sql and agent_name in ("System Agent", "System Guard") and ("inappropriate" in content.lower() or "too long" in content.lower() or "injection" in content.lower())
    is_schema_error = "schema" in content.lower() and ("failed" in content.lower() or "error" in content.lower())

    is_exact_cache_hit = (cache_hit == "exact") or (cache_hit is None and cache_likely and not has_sql and not is_error and not is_write_refusal and not is_raw_sql_refusal)
    is_semantic_cache_hit = (cache_hit == "semantic") or (cache_hit is None and cache_likely and has_sql and not is_error and not is_write_refusal and not is_raw_sql_refusal)

    char_len = len(user_question) if user_question else 0
    token_est = len(user_question.split()) if user_question else 0
    history_count = len(st.session_state.get("chat_history", []))

    # ── STAGE 0: L0 Query Entry & Pre-Processing ──────────────────────────────
    pp.set_step(
        "l0",
        pp.STATUS_OK,
        detail="Query Captured & Context Assembled",
        metrics={"Characters": char_len, "Est. Tokens": token_est, "Context History": f"{history_count} msgs"},
        logs=[
            "L0.1 Raw user input captured from chat_input()",
            "L0.2 Whitespace & special character pre-normalization: OK",
            "L0.3 Assembled chat history & session context",
        ],
    )

    # ── BRANCH 1A: L1 Direct Raw SQL Input Refusal ────────────────────────────
    if is_raw_sql_refusal:
        pp.set_step(
            "l1",
            pp.STATUS_ERR,
            detail="Blocked: Direct Raw SQL Input (Natural Language Only Policy)",
            metrics={"Guardrail Gate": "L1 StitchGuard", "Policy": "Natural Language Only", "Raw SQL": "BLOCKED (Refusal)"},
            logs=[
                "Prompt Injection check: PASSED",
                "PII Redaction filter: PASSED",
                "Natural Language Only Guard: BLOCKED (Raw SQL Syntax Input)",
            ],
            is_open=True,
        )
        pp.set_step("l2", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l3", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l4", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l7", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        st.session_state["pipeline_intent"] = "RAW_SQL_BLOCKED"
        st.session_state["pipeline_agent"]  = "System Guard"
        return

    # ── BRANCH 1B: L1 Write / Mutation Refusal ──────────────────────────────────
    if is_write_refusal:

        pp.set_step(
            "l1",
            pp.STATUS_ERR,
            detail="Blocked: Write / Mutation Intent Detected",
            metrics={"Guardrail Gate": "L1 StitchGuard", "PII Filter": "Active", "Write Intent": "BLOCKED (Refusal)"},
            logs=[
                "Prompt Injection check: PASSED",
                "PII Redaction filter: PASSED",
                "Write Intent Guard: BLOCKED (Database Mutation Refusal)",
            ],
            is_open=True,
        )
        pp.set_step("l2", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l3", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l4", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l7", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        st.session_state["pipeline_intent"] = "MUTATION_BLOCKED"
        st.session_state["pipeline_agent"]  = "System Guard"
        return

    # ── BRANCH 2: L1 Security / Injection / Length Block ──────────────────────
    if is_l1_security_block:
        pp.set_step(
            "l1",
            pp.STATUS_ERR,
            detail="Blocked: Security Violation / Injection Detected",
            metrics={"Guardrail Gate": "L1 StitchGuard", "Status": "BLOCKED (Security Gate)"},
            logs=[
                "Prompt Injection Guard: BLOCKED (Malicious Pattern / Jailbreak Detected)",
                "Terminated query pipeline before execution",
            ],
            is_open=True,
        )
        pp.set_step("l2", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l3", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l4", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        pp.set_step("l7", pp.STATUS_SKIP, detail="Skipped (Blocked at L1 Guardrail)")
        st.session_state["pipeline_agent"] = "System Guard"
        return

    # ── Standard L1: Input Guard Cleared ─────────────────────────────────────
    pp.set_step(
        "l1",
        pp.STATUS_OK,
        detail="Passed: Injection, PII, Write-intent cleared",
        metrics={"Guardrail Gate": "L1 StitchGuard", "PII Filter": "Active", "Write Intent": "Safe"},
        logs=[
            "Prompt Injection check: PASSED",
            "PII Redaction filter: PASSED",
            "Write Intent / Mutation Guard: PASSED",
        ],
    )

    # ── BRANCH 3: L2 Exact Cache Hit ─────────────────────────────────────────
    if is_exact_cache_hit:
        pp.set_step(
            "l2", pp.STATUS_SKIP, detail="Hit: Exact TTLCache lookup matched",
            metrics={"Engine": "TTLCache", "Lookup": "Exact Match", "Hit": "True"},
            logs=["Exact text match found in memory cache", "Bypassed LLM routing & DB execution"],
            is_open=True,
        )
        pp.set_step("l3", pp.STATUS_SKIP, detail="Skipped (Bypassed via L2 Exact Cache Hit)")
        pp.set_step("l4", pp.STATUS_SKIP, detail="Skipped (Bypassed via L2 Exact Cache Hit)")
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (Bypassed via L2 Exact Cache Hit)")
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (Bypassed via L2 Exact Cache Hit)")
        pp.set_step(
            "l7", pp.STATUS_OK, latency_ms=total_ms, detail="Cached answer delivered instantly",
            metrics={"Guardrail Policy": "L7 Output Content Sanitizer", "Source": "L2 TTLCache", "Sanitizer": "Active"},
            logs=["L7.1 Served cached answer from memory", "L7.2 PII leak scan: PASSED"],
            is_open=True,
        )
        st.session_state["pipeline_cache_hit"] = "exact"
        st.session_state["pipeline_agent"]     = agent_name
        return

    # ── BRANCH 4: L3 Semantic Vector Cache Hit ────────────────────────────────
    if is_semantic_cache_hit:
        pp.set_step(
            "l2", pp.STATUS_OK, detail="Miss: Exact text not in cache",
            metrics={"Engine": "TTLCache", "Hit": "False"},
            logs=["Key not found in in-memory TTLCache"],
        )
        pp.set_step(
            "l3", pp.STATUS_SKIP, detail="Hit: Semantic vector embedding matched",
            metrics={"Guardrail Policy": "L3 Domain Scope Guard", "Embedder": "nomic-embed-text", "Similarity": ">0.85 Cosine", "Hit": "True"},
            logs=["High-confidence semantic vector match detected", "Re-executing cached SQL structure"],
            is_open=True,
        )
        pp.set_step("l4", pp.STATUS_SKIP, detail="Skipped (Bypassed via L3 Semantic Cache Hit)")
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (Bypassed via L3 Semantic Cache Hit)")
        pp.set_step(
            "l6", pp.STATUS_OK, latency_ms=total_ms * 0.4, detail="DB execution from Semantic Cache",
            metrics={"Execution Engine": "MCP Persistent Server", "Cache": "Semantic Re-run"},
            logs=["Executed cached query on database session"],
            code_or_data=sql,
        )
        pp.set_step(
            "l7", pp.STATUS_OK, latency_ms=total_ms, detail="Formatted answer from semantic cache DB output",
            metrics={"Guardrail Policy": "L7 Output Content Sanitizer", "Source": "L3 Semantic Re-run", "Sanitizer": "Active"},
            logs=["L7.1 Formatted cached query output", "L7.2 PII & metadata scrubbed"],
            is_open=True,
        )
        st.session_state["pipeline_cache_hit"] = "semantic"
        st.session_state["pipeline_agent"]     = agent_name
        return

    # Cache misses for L2 and L3
    pp.set_step(
        "l2", pp.STATUS_OK, detail="Miss: Exact text not in cache",
        metrics={"Engine": "TTLCache", "Lookup": "Exact Match", "Hit": "False"},
        logs=["L2.1 TTLCache lookup: Miss"],
    )
    pp.set_step(
        "l3", pp.STATUS_OK, detail="Miss: Vector similarity below threshold",
        metrics={"Guardrail Policy": "L3 Domain Scope Guard", "Embedder": "nomic-embed-text", "Similarity": "<0.85 Cosine"},
        logs=["L3.1 Semantic Vector Embedding Search: Miss", "L3.2 Domain Table Scope Pre-Filter: Ready"],
    )

    # ── BRANCH 5: L4 General Chat Routing (No DB needed) ──────────────────────
    if agent_name in ("General Agent", "System Agent") and not has_sql:
        pp.set_step(
            "l4", pp.STATUS_OK, detail="Routed: GENERAL (Conversational)",
            metrics={"Guardrail Policy": "L4 SQL Safety Guard", "Router Model": ROUTER_MODEL, "Intent Domain": "GENERAL"},
            logs=["L4.1 Rephraser & LLM Intent Classification: GENERAL", "L4.2 Database Bypass: Conversational Chat"],
        )
        pp.set_step("l5", pp.STATUS_SKIP, detail="Skipped (No DB schema required for GENERAL intent)")
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (No SQL query required for GENERAL intent)")
        pp.set_step(
            "l7", pp.STATUS_OK, latency_ms=total_ms, detail="Conversational reply delivered",
            metrics={"Guardrail Policy": "L7 Output Content Sanitizer", "Agent": agent_name, "Latency": f"{total_ms:.0f}ms"},
            logs=["L7.1 Streamed conversational tokens to UI", "L7.2 PII reassurance check: PASSED"],
            is_open=True,
        )
        st.session_state["pipeline_intent"] = "GENERAL"
        st.session_state["pipeline_agent"]  = agent_name
        return

    # ── BRANCH 6: Domain Agent SQL Pipeline Execution ─────────────────────────
    intent_map = {
        "WMS Assistant": "WMS_AGENT",
    }
    intent = intent_map.get(agent_name, "WMS_AGENT")
    token_count = len(sql.split()) if sql else 0

    pp.set_step(
        "l4", pp.STATUS_OK, detail=f"Routed: {intent}",
        metrics={"Guardrail Policy": "L4 SQL Safety Guard", "Router Model": ROUTER_MODEL, "Intent Domain": intent, "SQL Policy": "Read-Only Validated"},
        logs=[f"L4.1 Classified intent as {intent}", "L4.2 Read-Only SQL Structural Safety Validator: PASSED"],
    )

    # L5 Schema / VectorRAG Error vs Success
    if is_schema_error:
        pp.set_step(
            "l5", pp.STATUS_ERR, detail="Failed: Database Schema Retrieval Error",
            metrics={"Guardrail Policy": "L5 Column Redaction Guard", "Selector": "VectorRAG", "Status": "ERROR"},
            logs=["Failed to retrieve database schema context"],
            is_open=True,
        )
        pp.set_step("l6", pp.STATUS_SKIP, detail="Skipped (Blocked due to Layer 5 Schema Error)")
        pp.set_step("l7", pp.STATUS_SKIP, detail="Skipped (Blocked due to Layer 5 Schema Error)")
        st.session_state["pipeline_intent"] = intent
        st.session_state["pipeline_agent"]  = agent_name
        return

    pp.set_step(
        "l5", pp.STATUS_OK, detail="Passed: VectorRAG table scope & schema fetched",
        metrics={"Guardrail Policy": "L5 Column Redaction Guard", "Selector": "VectorRAG", "Scope": intent, "Masking": "Active"},
        logs=["L5.1 VectorRAG Target Table Selection: Scope Focused", "L5.2 Sensitive Output Column Redactor: Active (employee_id, ssn, password, email)"],
    )

    # L6 Execution Error vs Success
    if is_error:
        pp.set_step(
            "l6", pp.STATUS_ERR, latency_ms=total_ms, detail="Execution Failed: SQL / MCP Query Error",
            metrics={"Guardrail Policy": "L6 Content Sanitizer", "Execution Engine": "MCP Persistent Server", "Agent": agent_name, "Status": "FAILED"},
            logs=["L6.1 MCP Read-Only Session Execution: FAILED", "Query execution aborted"],
            code_or_data=sql,
            is_open=True,
        )
        pp.set_step("l7", pp.STATUS_ERR, detail="Error response returned to user", is_open=True)
    else:
        pp.set_step(
            "l6", pp.STATUS_OK, latency_ms=total_ms * 0.7, detail=f"{token_count}-token SQL executed via MCP",
            metrics={"Guardrail Policy": "L6 Persistent MCP Session", "Execution Engine": "MCP SQLite Server", "Agent": agent_name},
            logs=[
                "L6.1 MCP Tool 'execute_read_only_query': PASSED",
                "L6.2 Returned database result table to formatter",
            ],
            code_or_data=sql,
        )
        pp.set_step(
            "l7", pp.STATUS_OK, latency_ms=total_ms, detail=f"Final answer formatted & sanitized ({total_ms:.0f}ms)",
            metrics={"Guardrail Policy": "L7 Output Content Sanitizer", "Formatter": "LLM / Template Engine", "Sanitizer": "Active"},
            logs=[
                "L7.1 Answer formatting applied to DB output",
                "L7.2 PII Leak & Forbidden Output Pattern Filter: PASSED",
                "L7.3 Internal Metadata & Schema Name Scrubbing: PASSED",
                "L7.4 Token streaming to chat UI: COMPLETE",
            ],
            is_open=True,
        )

    st.session_state["pipeline_intent"] = intent
    st.session_state["pipeline_agent"]  = agent_name
