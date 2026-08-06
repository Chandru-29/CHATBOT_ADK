"""
pipeline_panel.py — Renders the slim left pipeline-flow panel.

Shows all 6 backend pipeline layers (L1 -> L6) as a vertical step-list with
live status icons and the last resolved intent/agent.

State keys written by chat_window.py (before/during/after a query):
  st.session_state["pipeline_status"]  — dict keyed by step id
  st.session_state["pipeline_intent"]  — str or None
  st.session_state["pipeline_agent"]   — str or None
  st.session_state["pipeline_cache_hit"] — "exact" | "semantic" | None

Design note: All rendering uses small, self-contained inline-styled divs via
st.markdown() — NO <style> blocks or multi-level nested HTML — so Streamlit
columns stay isolated and no HTML leaks into adjacent columns.
"""


# -- MODULE TAG: Streamlit UI Pipeline Flow Panel --
import streamlit as st

# -- Step definitions: (id, short label, emoji icon, short description) --------
PIPELINE_STEPS = [
    ("l0", "Query Entry & Pre-Processing", "L0 Input",    "Input capture · Token estimation · History context"),
    ("l1", "Input Guard Gate",             "L1 Guard",    "Injection · PII · Write-intent filter"),
    ("l2", "Exact Cache Lookup",            "L2 Cache",    "TTL key-value fast path"),
    ("l3", "Semantic Cache & Scope",        "L3 Scope",    "Cosine similarity + Authorized domain scope"),
    ("l4", "Intent Router & SQL Safety",    "L4 Safety",   "LLM router + Read-only SQL validator"),
    ("l5", "RAG & Column Redactor",         "L5 Redact",   "VectorRAG scope + Sensitive column masking"),
    ("l6", "MCP Session Execution",         "L6 MCP",      "Read-only SQLite persistent session execution"),
    ("l7", "Answer Formatter & Sanitizer",  "L7 Output",   "Result formatting + PII audit + Streaming delivery"),
]


# -- Status tokens -------------------------------------------------------------
STATUS_IDLE   = "idle"    # grey  – not yet run
STATUS_SKIP   = "skip"    # blue  – bypassed (cache hit)
STATUS_OK     = "ok"      # green – completed OK
STATUS_ERR    = "error"   # red   – failed
STATUS_ACTIVE = "active"  # amber – in-flight

# -- Per-status display config: (dot_char, dot_bg, dot_border, label_color) ----
_STATUS_CFG = {
    STATUS_IDLE:   ("·",  "#f8fafc", "#e2e8f0", "#64748b"),
    STATUS_SKIP:   ("⇢",  "#eff6ff", "#93c5fd", "#2563eb"),
    STATUS_OK:     ("✓",  "#f0fdf4", "#a7f3d0", "#10b981"),
    STATUS_ERR:    ("✗",  "#fef2f2", "#fca5a5", "#dc2626"),
    STATUS_ACTIVE: ("◎",  "#e0f2fe", "#38bdf8", "#0284c7"),
}


# -- Intent display: (label, fg, bg) ------------------------------------------
_INTENT_CFG = {
    "WMS_AGENT":    ("WMS",     "#2563eb", "#eff6ff"),
    "GENERAL":      ("General", "#6b7280", "#f9fafb"),
}


# -- Session state helpers -----------------------------------------------------

import html

def _idle_state() -> dict:
    return {
        s[0]: {
            "status":       STATUS_IDLE,
            "latency_ms":   None,
            "detail":       "",
            "metrics":      {},
            "logs":         [],
            "code_or_data": None,
            "is_open":      False,
        }
        for s in PIPELINE_STEPS
    }


def init_pipeline_state() -> None:
    """Ensure all pipeline session keys exist on first load."""
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = _idle_state()
    if "pipeline_intent" not in st.session_state:
        st.session_state["pipeline_intent"] = None
    if "pipeline_agent" not in st.session_state:
        st.session_state["pipeline_agent"] = None
    if "pipeline_cache_hit" not in st.session_state:
        st.session_state["pipeline_cache_hit"] = None


def reset_pipeline() -> None:
    """Reset all steps to idle before a new query starts."""
    st.session_state["pipeline_status"]   = _idle_state()
    st.session_state["pipeline_intent"]   = None
    st.session_state["pipeline_agent"]    = None
    st.session_state["pipeline_cache_hit"] = None


def set_step(
    step_id: str,
    status: str,
    latency_ms: float = None,
    detail: str = "",
    metrics: dict = None,
    logs: list = None,
    code_or_data: str = None,
    is_open: bool = False,
) -> None:
    """Update one pipeline step. Called from chat_window.py."""
    if "pipeline_status" not in st.session_state:
        init_pipeline_state()
    st.session_state["pipeline_status"][step_id] = {
        "status":       status,
        "latency_ms":   latency_ms,
        "detail":       detail,
        "metrics":      metrics or {},
        "logs":         logs or [],
        "code_or_data": code_or_data,
        "is_open":      is_open or (status in (STATUS_ACTIVE, STATUS_ERR)),
    }


# -- Rendering -----------------------------------------------------------------

def _fmt_latency(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"


def _build_step_html(step_id: str, badge: str, label: str, desc: str, info: dict, is_last: bool = False) -> str:
    """Return an interactive accordion HTML string for a single pipeline step layer."""
    status       = info.get("status", STATUS_IDLE)
    latency_ms   = info.get("latency_ms")
    detail       = info.get("detail", "")
    metrics      = info.get("metrics", {})
    logs         = info.get("logs", [])
    code_or_data = info.get("code_or_data")
    is_open      = info.get("is_open", status in (STATUS_ACTIVE, STATUS_ERR))

    dot_char, dot_bg, dot_border, label_color = _STATUS_CFG.get(status, _STATUS_CFG[STATUS_IDLE])
    latency_str = _fmt_latency(latency_ms)
    anim_style  = "animation:pp-pulse 1.2s ease-in-out infinite;" if status == STATUS_ACTIVE else ""

    latency_html = (
        f'<span class="pd-latency">&#x23F1;{latency_str}</span>'
        if latency_str else ""
    )

    open_attr = ' open="open"' if is_open else ""
    connector_html = '<div class="pd-connector"></div>' if not is_last else ''

    # StitchGuard badge tag for security enforcement layers
    guardrail_badge = (
        '<span class="pd-guardrail-badge">🛡️ StitchGuard</span>'
        if step_id in ("l1", "l3", "l4", "l5", "l6", "l7") else ""
    )

    # Build accordion body sections
    body_parts = []
    if detail:
        body_parts.append(f'<div class="pd-detail-summary">{html.escape(detail)}</div>')

    if metrics:
        chip_htmls = [
            f'<div class="pd-metric-chip"><span class="pd-metric-label">{html.escape(str(k))}:</span> {html.escape(str(v))}</div>'
            for k, v in metrics.items()
        ]
        body_parts.append(f'<div class="pd-metrics-grid">{"".join(chip_htmls)}</div>')

    if logs:
        log_items = [f'<li class="pd-log-item">&#x25B8; {html.escape(str(l))}</li>' for l in logs]
        body_parts.append(f'<ul class="pd-log-list">{"".join(log_items)}</ul>')

    if code_or_data:
        body_parts.append(f'<div class="pd-code-block">{html.escape(str(code_or_data))}</div>')

    body_html = (
        f'<div class="pd-accordion-body">{"".join(body_parts)}</div>'
        if body_parts else ""
    )

    if status == STATUS_ACTIVE:
        accordion_cls = "pd-accordion pd-accordion-active"
        dot_html = '<div class="pd-dot pd-dot-active"><div class="pd-spinner"></div></div>'
        desc_html = '<div class="pd-step-desc pd-step-desc-active">&#x26A1; Executing stage...</div>'
    else:
        accordion_cls = "pd-accordion"
        dot_html = f'<div class="pd-dot" style="background:{dot_bg};border:1.5px solid {dot_border};color:{label_color};">{dot_char}</div>'
        desc_html = f'<div class="pd-step-desc">{html.escape(desc)}</div>'

    return (
        f'<details class="{accordion_cls}"{open_attr}>'
        f'<summary class="pd-summary">'
        f'<div class="pd-summary-left">'
        f'{dot_html}'
        f'{connector_html}'
        f'</div>'
        f'<div class="pd-summary-content">'
        f'<div class="pd-step-title">{badge} {html.escape(label)}{latency_html} {guardrail_badge}</div>'
        f'{desc_html}'
        f'</div>'
        f'<div class="pd-chevron">&#x25B6;</div>'
        f'</summary>'
        f'{body_html}'
        f'</details>'
    )


def _render_step(step_id: str, badge: str, label: str, desc: str, is_last: bool = False) -> None:
    """Render a single pipeline step row as an interactive accordion using st.markdown."""
    info = st.session_state["pipeline_status"].get(
        step_id,
        {"status": STATUS_IDLE, "latency_ms": None, "detail": "", "metrics": {}, "logs": [], "code_or_data": None}
    )
    is_active = info.get("status") == STATUS_ACTIVE
    html_content = _build_step_html(step_id, badge, label, desc, info, is_last=is_last)

    if is_active:
        html_content = (
            "<style>@keyframes pp-pulse{"
            "0%,100%{opacity:1;transform:scale(1)}"
            "50%{opacity:.45;transform:scale(.8)}"
            "}</style>"
        ) + html_content

    st.markdown(html_content, unsafe_allow_html=True)


def _render_footer(intent: str | None, agent: str | None, cache_hit: str | None) -> None:
    """Render the intent/agent/cache summary row at the bottom of the panel."""
    if not any([intent, agent, cache_hit]):
        return

    rows = []

    if cache_hit:
        fg  = "#2563eb" if cache_hit == "exact" else "#7c3aed"
        bg  = "#eff6ff" if cache_hit == "exact" else "#f5f3ff"
        lbl = f"⚡ {cache_hit.capitalize()} cache"
        rows.append(
            f"<span style='background:{bg};border:1px solid {fg}44;"
            f"border-radius:4px;padding:2px 6px;font-size:9.5px;"
            f"color:{fg};font-weight:600;'>{lbl}</span>"
        )

    if intent and intent in _INTENT_CFG:
        lbl, fg, bg = _INTENT_CFG[intent]
        rows.append(
            f"<span style='background:{bg};border:1px solid {fg}44;"
            f"border-radius:4px;padding:2px 6px;font-size:9.5px;"
            f"color:{fg};font-weight:600;'>{lbl}</span>"
        )

    if agent:
        rows.append(
            f"<span style='font-size:9.5px;color:#71717a;'>🤖 {agent}</span>"
        )

    if rows:
        inner = "&nbsp; ".join(rows)
        st.markdown(
            f"<div style='padding-top:4px;line-height:1.8;'>{inner}</div>",
            unsafe_allow_html=True,
        )


def render_pipeline_panel() -> None:
    """
    Draw the pipeline flow panel inside a Streamlit column.
    Uses only native st.* calls + simple, flat, self-contained HTML per step.
    No <style> block at panel level — avoids HTML leaking into sibling columns.
    """
    init_pipeline_state()

    # -- Panel header ----------------------------------------------------------
    st.markdown(
        "<p style='margin:0 0 10px 0;font-size:10px;font-weight:700;"
        "color:#71717a;letter-spacing:.07em;text-transform:uppercase;'>"
        "⚙ Pipeline Flow</p>",
        unsafe_allow_html=True,
    )

    # -- One row per step ------------------------------------------------------
    steps = PIPELINE_STEPS
    for i, (step_id, badge, label, desc) in enumerate(steps):
        _render_step(step_id, badge, label, desc, is_last=(i == len(steps) - 1))

    # -- Footer: intent + agent + cache badge ----------------------------------
    _render_footer(
        intent    = st.session_state.get("pipeline_intent"),
        agent     = st.session_state.get("pipeline_agent"),
        cache_hit = st.session_state.get("pipeline_cache_hit"),
    )


def _build_footer_html(intent: str | None, agent: str | None, cache_hit: str | None) -> str:
    """Return an HTML string for the intent/agent/cache summary row."""
    parts = []

    if cache_hit:
        fg  = "#2563eb" if cache_hit == "exact" else "#7c3aed"
        bg  = "#eff6ff" if cache_hit == "exact" else "#f5f3ff"
        lbl = f"&#x26A1; {cache_hit.capitalize()} cache"
        parts.append(
            f'<div style="background:{bg};border:1px solid {fg}44;border-radius:5px;'
            f'padding:4px 9px;font-size:10px;color:{fg};font-weight:600;margin-bottom:5px;">{lbl}</div>'
        )

    if intent and intent in _INTENT_CFG:
        lbl, fg, bg = _INTENT_CFG[intent]
        parts.append(
            f'<div style="background:{bg};border:1px solid {fg}44;border-radius:5px;'
            f'padding:4px 9px;font-size:10px;color:{fg};font-weight:600;margin-bottom:5px;">{lbl}</div>'
        )

    if agent:
        parts.append(
            f'<div style="font-size:10px;color:#71717a;padding:2px 0;">&#x1F916; {agent}</div>'
        )

    if not parts:
        return ""

    return (
        '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #f4f4f5;">'
        + "".join(parts)
        + "</div>"
    )


def build_pipeline_html(
    state:     dict,
    intent:    str | None,
    agent:     str | None,
    cache_hit: str | None,
) -> str:
    """
    Return the complete pipeline panel body as an HTML string.
    Called by right_drawer.py to embed content in the fixed-position overlay.
    """
    parts = []
    steps = PIPELINE_STEPS
    for i, (step_id, badge, label, desc) in enumerate(steps):
        info = state.get(step_id, {"status": STATUS_IDLE, "latency_ms": None, "detail": ""})
        step_html = _build_step_html(step_id, badge, label, desc, info, is_last=(i == len(steps) - 1))
        parts.append(step_html)

    parts.append(_build_footer_html(intent, agent, cache_hit))
    return "\n".join(parts)
