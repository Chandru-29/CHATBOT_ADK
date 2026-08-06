"""
right_drawer.py — Fixed-position right-side sliding pipeline drawer.

Uses st.components.v1.html() (a real iframe) to execute JavaScript that
injects the drawer directly into the parent Streamlit document. This is the
only fully reliable approach in Streamlit because:

  1. st.markdown(unsafe_allow_html=True) — scripts stripped/not re-executed
     on reruns; onclick attrs can't cross React component boundaries reliably.

  2. st.components.v1.html() — runs in a sandboxed iframe from the SAME
     origin as the main app (localhost), so window.parent.document access
     works and addEventListener fires correctly every time.

The pipeline body HTML is JSON-encoded before embedding so no escaping issues
can arise from user-supplied SQL or agent text in the panel content.
"""


# ── MODULE TAG: Streamlit UI Right Drawer (Pipeline) ──
import json
import streamlit as st
import streamlit.components.v1 as components

from ui.pipeline_panel import (
    init_pipeline_state,
    _idle_state,
    build_pipeline_html,
)


# ── Drawer CSS (injected once into parent <head>) ──────────────────────────────
_CSS = """
@keyframes pp-pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.75); }
}
@keyframes pd-pulse-glow {
  0%, 100% {
    box-shadow: 0 0 6px rgba(56, 189, 248, 0.4), inset 0 0 4px rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.7) !important;
  }
  50% {
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.8), inset 0 0 8px rgba(56, 189, 248, 0.25);
    border-color: #0284c7 !important;
  }
}
@keyframes pd-spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.pd-accordion-active {
  animation: pd-pulse-glow 1.5s ease-in-out infinite !important;
  background: rgba(240, 249, 255, 0.95) !important;
  border-width: 1.5px !important;
}
.pd-dot-active {
  background: #e0f2fe !important;
  border: 1.5px solid #38bdf8 !important;
}
.pd-spinner {
  width: 11px;
  height: 11px;
  border: 2px solid rgba(56, 189, 248, 0.3);
  border-top: 2px solid #0284c7;
  border-radius: 50%;
  animation: pd-spin 0.75s linear infinite;
  display: inline-block;
  vertical-align: middle;
}
.pd-step-desc-active {
  color: #0284c7 !important;
  font-weight: 600 !important;
}

.pd-connector {
  width: 1.5px; flex: 1; min-height: 10px;
  background: #e4e4e7; margin: 2px 0;
}
#pd-root, #pd-root * {
  pointer-events: auto !important;
}
#pd-overlay {
  position: fixed; inset: 0;
  background: transparent !important;
  z-index: 9998; display: none;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  pointer-events: none !important;
}
#pd-overlay.pd-vis { display: block; }
#pd-toggle {
  position: fixed; right: 18px; top: 62px;
  z-index: 2147483647 !important;
  pointer-events: auto !important;
  width: 38px; height: 38px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.90);
  border: 1px solid rgba(228, 228, 231, 0.9);
  box-shadow: 0 4px 16px rgba(0,0,0,.08);
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: #18181b;
  font-family: sans-serif;
  transition: background .15s, box-shadow .15s, transform .12s;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
#pd-toggle:hover {
  background: #ffffff;
  box-shadow: 0 6px 20px rgba(0,0,0,.12);
  transform: scale(1.06);
}
#pd-toggle.pd-on {
  background: #18181b; color: #fff;
  border-color: #18181b;
  box-shadow: 0 4px 20px rgba(0,0,0,.25);
}
#pd-drawer {
  position: fixed; top: 0; right: -310px;
  width: 290px; height: 100vh;
  background: rgba(255, 255, 255, 0.88) !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  border-left: 1px solid rgba(228, 228, 231, 0.8) !important;
  z-index: 2147483646 !important;
  pointer-events: auto !important;
  transition: right .28s cubic-bezier(.4,0,.2,1);
  overflow-y: auto; overflow-x: hidden;
  box-shadow: -10px 0 30px rgba(0, 0, 0, 0.08) !important;
  font-family: 'Inter', ui-sans-serif, sans-serif;
  color: #18181b;
}
#pd-drawer.pd-open { right: 0; }
.pd-inner { padding: 58px 16px 28px; }
.pd-hdr {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px; padding-bottom: 10px;
  border-bottom: 1px solid #e4e4e7;
}
.pd-hdr-title {
  font-size: 10px; font-weight: 700;
  color: #71717a; letter-spacing: .08em; text-transform: uppercase;
}
.pd-x {
  background: none; border: none; cursor: pointer;
  color: #a1a1aa; font-size: 18px; line-height: 1;
  padding: 3px 5px; border-radius: 5px;
  transition: color .12s, background .12s;
  pointer-events: auto !important;
}
.pd-x:hover { color: #18181b; background: #f4f4f5; }

/* Interactive Step Accordions */
details.pd-accordion {
  border: 1px solid #e4e4e7;
  border-radius: 8px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.80);
  overflow: hidden;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
  pointer-events: auto !important;
}
details.pd-accordion[open] {
  border-color: #d4d4d8;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
summary.pd-summary {
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  cursor: pointer;
  user-select: none;
  font-family: inherit;
  transition: background 0.12s ease;
  pointer-events: auto !important;
}
summary.pd-summary::-webkit-details-marker,
summary.pd-summary::marker {
  display: none;
  content: "";
}
summary.pd-summary:hover {
  background: rgba(248, 250, 252, 0.90);
}
.pd-summary-left {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex-shrink: 0;
}
.pd-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  flex-shrink: 0;
}
.pd-summary-content {
  flex: 1;
  min-width: 0;
}
.pd-step-title {
  font-size: 11.5px;
  font-weight: 600;
  color: #18181b;
  line-height: 1.3;
}
.pd-step-desc {
  font-size: 9.5px;
  color: #71717a;
  line-height: 1.3;
  margin-top: 1px;
}
.pd-latency {
  font-size: 9px;
  color: #a1a1aa;
  font-family: monospace;
  margin-left: 4px;
}
.pd-chevron {
  font-size: 8px;
  color: #a1a1aa;
  transition: transform 0.2s ease;
  flex-shrink: 0;
  margin-left: 2px;
}
details[open] > summary .pd-chevron {
  transform: rotate(90deg);
}
.pd-accordion-body {
  padding: 8px 10px 10px 10px;
  border-top: 1px solid #f4f4f5;
  background: rgba(250, 250, 250, 0.85);
  font-size: 10px;
  color: #3f3f46;
  pointer-events: auto !important;
}
.pd-detail-summary {
  font-size: 9.5px;
  color: #52525b;
  line-height: 1.35;
  word-break: break-word;
}
.pd-metrics-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
.pd-metric-chip {
  background: #ffffff;
  border: 1px solid #e4e4e7;
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 9px;
  color: #52525b;
}
.pd-metric-label {
  font-weight: 600;
  color: #71717a;
}
.pd-log-list {
  margin: 6px 0 0 0;
  padding: 0;
  list-style: none;
}
.pd-log-item {
  font-size: 9px;
  color: #71717a;
  padding: 1px 0;
}
.pd-code-block {
  margin-top: 6px;
  padding: 6px 8px;
  background: #18181b;
  color: #38bdf8;
  border-radius: 5px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 9px;
  line-height: 1.4;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.pd-guardrail-badge {
  font-size: 8.5px;
  font-weight: 600;
  color: #047857;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  border-radius: 4px;
  padding: 1px 4px;
  margin-left: 4px;
  vertical-align: middle;
}
"""


def _build_drawer_html(pipeline_body: str) -> str:
    """Return the inner HTML of the drawer panel."""
    return (
        '<div id="pd-overlay"></div>'
        '<button id="pd-toggle" title="Toggle Pipeline">&#9881;</button>'
        '<div id="pd-drawer">'
          '<div class="pd-inner">'
            '<div class="pd-hdr">'
              '<span class="pd-hdr-title">&#9881;&nbsp;Pipeline Flow</span>'
              '<button class="pd-x" title="Close">&#10005;</button>'
            '</div>'
            + pipeline_body +
          '</div>'
        '</div>'
    )


def render_right_drawer() -> None:
    """
    Render the right-side pipeline drawer.

    Injects a <style> tag and the drawer HTML into the parent Streamlit
    document via window.parent.document from within a components.v1.html()
    iframe. Event listeners are wired with addEventListener (not onclick attrs)
    so they survive DOM updates.
    """
    init_pipeline_state()

    state     = st.session_state.get("pipeline_status", _idle_state())
    intent    = st.session_state.get("pipeline_intent")
    agent     = st.session_state.get("pipeline_agent")
    cache_hit = st.session_state.get("pipeline_cache_hit")

    pipeline_body = build_pipeline_html(state, intent, agent, cache_hit)
    drawer_html   = _build_drawer_html(pipeline_body)

    # JSON-encode so any special chars (backticks, quotes, </) are safe in JS
    css_js  = json.dumps(_CSS)
    html_js = json.dumps(drawer_html)
    # Prevent </script> from ending our script block prematurely
    css_js  = css_js.replace("</", "<\\/")
    html_js = html_js.replace("</", "<\\/")

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body>
<script>
(function() {{
  try {{
    var P = window.parent.document;

    // ── Inject CSS once into parent <head> ───────────────────────────────
    if (!P.getElementById('pd-style')) {{
      var st = P.createElement('style');
      st.id = 'pd-style';
      st.textContent = {css_js};
      P.head.appendChild(st);
    }}

    // ── Remember if drawer was already open ──────────────────────────────
    var wasOpen = false;
    var old = P.getElementById('pd-drawer');
    if (old) wasOpen = old.classList.contains('pd-open');

    // ── Upsert root container in parent <body> ───────────────────────────
    var root = P.getElementById('pd-root');
    if (!root) {{
      root = P.createElement('div');
      root.id = 'pd-root';
      P.body.appendChild(root);
    }}
    root.innerHTML = {html_js};

    // ── Grab elements ────────────────────────────────────────────────────
    var drawer  = P.getElementById('pd-drawer');
    var toggle  = P.getElementById('pd-toggle');
    var overlay = P.getElementById('pd-overlay');
    var closeX  = P.querySelector('#pd-drawer .pd-x');

    // ── Helper functions (defined fresh each inject) ─────────────────────
    function pdOpen(e)  {{
      if (e) {{ e.preventDefault(); e.stopPropagation(); }}
      drawer.classList.add('pd-open');
      toggle.classList.add('pd-on');
      overlay.classList.add('pd-vis');
    }}
    function pdClose(e) {{
      if (e) {{ e.preventDefault(); e.stopPropagation(); }}
      drawer.classList.remove('pd-open');
      toggle.classList.remove('pd-on');
      overlay.classList.remove('pd-vis');
    }}
    function pdToggle(e) {{
      if (e) {{ e.preventDefault(); e.stopPropagation(); }}
      drawer.classList.contains('pd-open') ? pdClose(e) : pdOpen(e);
    }}

    // ── Wire event listeners (fresh after innerHTML replace) ─────────────
    if (toggle)  toggle.addEventListener('click',  pdToggle, true);
    if (overlay) overlay.addEventListener('click',  pdClose, true);
    if (closeX)  closeX.addEventListener('click',   pdClose, true);

    // ── Restore open/closed state ────────────────────────────────────────
    if (wasOpen) pdOpen();

  }} catch(err) {{
    console.warn('[PipelineDrawer]', err);
  }}
}})();
</script>
</body></html>"""

    components.html(page, height=0, scrolling=False)
