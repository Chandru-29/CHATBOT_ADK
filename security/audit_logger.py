"""
audit_logger.py — Staged Pipeline Trace Logger.

Emits structured, stage-numbered logs for every request lifecycle phase:
  [1] CLIENT & RUNNER ENTRY
  [2] MIDDLEWARE & GUARDRAILS (Layer 1-2)
  [3] ROUTER AGENT & REPHRASING
  [4] WMS SQL AGENT & TOOL EXECUTION (Layer 3-4)
  [5] OUTPUT FORMATTING & SANITIZATION (Layer 5-6)
"""

# ── MODULE TAG: Security Audit Logger ──
import uuid
from typing import List, Optional


def generate_req_id() -> str:
    """Generate a unique 6-character hex request ID (e.g. req_9d494e)."""
    return f"req_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Client & Runner Entry
# ─────────────────────────────────────────────────────────────────────────────

def log_incoming_request(log, req_id: str, query: str, stream: bool = False):
    """Log Stage 1: incoming request received."""
    log.info(f"[1. CLIENT & RUNNER ENTRY]")
    log.info(f"  └── Received Query: \"{query}\" | req_id={req_id} | stream={stream}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Middleware & Guardrails (L1 + L2 cache)
# ─────────────────────────────────────────────────────────────────────────────

def log_l1_audit(log, req_id: str, is_safe: bool, has_write: bool,
                 details_override: Optional[str] = None):
    """Log Stage 2 L1: input sanitation result."""
    if not is_safe or has_write:
        status  = "BLOCKED"
        details = details_override or (
            "Write/DDL intent detected — read-only policy enforced"
            if has_write else
            "Prompt injection or length limit violated"
        )
    else:
        status  = "PASS"
        details = details_override or "Input sanitized. No prompt injection or PII detected."

    log.info(f"[2. MIDDLEWARE & GUARDRAILS (Layer 1-2)]")
    log.info(f"  └── L1 Security Check: {details}")
    if status == "BLOCKED":
        log.warning(f"  └── ⛔ Request BLOCKED by L1 | req_id={req_id}")


def log_l2_cache(log, hit: bool, hit_source: str = ""):
    """Log Stage 2 L2: cache hit/miss."""
    if hit:
        log.info(f"  └── L2 Semantic Cache: Cache HIT ({hit_source.upper()}). Returning cached result.")
    else:
        log.info(f"  └── L2 Semantic Cache: Cache miss. Proceeding to Agent workflow.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Router Agent & Rephrasing
# ─────────────────────────────────────────────────────────────────────────────

def log_router_decision(log, original_q: str, rephrased_q: str, intent: str,
                        confidence: float, was_rephrased: bool):
    """Log Stage 3: rephraser + router classification."""
    log.info(f"[3. ROUTER AGENT & REPHRASING (Gemini 2.5 Flash)]")
    if was_rephrased:
        log.info(f"  └── Query Rephraser: Follow-up resolved → \"{rephrased_q}\"")
    else:
        log.info(f"  └── Query Rephraser: Self-contained query — no rephrasing needed.")
    log.info(f"  └── Router Decision: Intent classified as {intent} (Confidence: {confidence:.2f}).")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — WMS SQL Agent & Tool Execution (L3 + L4)
# ─────────────────────────────────────────────────────────────────────────────

def log_rag_selection(log, tables: List[str], sim_score: float, threshold: float):
    """Log Stage 4: VectorRAG table grounding result."""
    log.info(f"[4. WMS SQL AGENT & TOOL EXECUTION (Layer 3-4)]")
    tables_str = str(tables) if tables else "all domain tables (fallback)"
    log.info(f"  └── VectorRAG Table Selector: Grounded tables → {tables_str} (Similarity: {sim_score:.2f} > {threshold:.2f})")


def log_tool_execution(log, sql: str):
    """Log Stage 4: SQL tool invocation."""
    sql_preview = " ".join(sql.split())[:200]
    log.info(f"  └── Tool Execution: Invoking ExecuteReadOnlySQL → {sql_preview}")


def log_l3_l4_guardrails(log, l3_ok: bool, l4_ok: bool,
                          l3_detail: str = "", l4_detail: str = ""):
    """Log Stage 4: L3/L4 guardrail results."""
    l3_msg = l3_detail or ("Table scope validated." if l3_ok else "Table scope violation.")
    l4_msg = l4_detail or ("Read-only check passed. No write/drop intent found." if l4_ok else "Write/DDL intent detected.")
    l3_icon = "✅" if l3_ok else "❌"
    l4_icon = "✅" if l4_ok else "❌"
    log.info(f"  └── L3 Table Scope Guard: {l3_icon} {l3_msg}")
    log.info(f"  └── L4 Read-Only Guard: {l4_icon} {l4_msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Output Formatting & Sanitization (L5 + L6)
# ─────────────────────────────────────────────────────────────────────────────

def log_output_pipeline(log, req_id: str, intent: str, sql: Optional[str],
                        duration_ms: float, status: str = "PASSED",
                        l5_detail: str = "", l6_detail: str = ""):
    """Log Stage 5: L5 redaction, L6 safety, response dispatch."""
    total_sec = duration_ms / 1000.0
    l5_msg = l5_detail or "No restricted/sensitive columns exposed."
    l6_msg = l6_detail or "Output content safety checked."

    log.info(f"[5. OUTPUT FORMATTING & SANITIZATION (Layer 5-6)]")
    log.info(f"  └── L5 Sensitive Column Redaction: {l5_msg}")
    log.info(f"  └── L6 Content Safety: {l6_msg}")

    if sql:
        log.info(f"  └── Deterministic Formatter: Converted SQL result rows into clean Markdown table.")
    else:
        log.info(f"  └── Deterministic Formatter: Natural language response composed (no SQL rows).")

    log.info(
        f"  └── Response Sent to Client | "
        f"Latency: {total_sec:.2f}s | "
        f"Intent: {intent} | "
        f"Status: {status} | "
        f"req_id={req_id}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compatibility shims (keep existing call-sites compiling)
# ─────────────────────────────────────────────────────────────────────────────

def log_execution_pipeline_trace(log, trace: dict):
    """Legacy shim — no-op (replaced by per-stage calls in runner.py)."""
    pass


def log_pipeline_terminated(log, req_id: str, total_ms: float, http_status: str = "200 OK"):
    """Legacy shim — kept for any external callers."""
    pass


def log_audit_tree(log, req_id: str, intent: str, tables: List[str],
                   duration_ms: float, status: str = "PASSED",
                   l3_details: Optional[str] = None,
                   l4_details: Optional[str] = None,
                   l5_details: Optional[str] = None,
                   l6_details: Optional[str] = None,
                   pipeline_info: str = "SQL generated & executed successfully"):
    """Legacy shim — calls the new Stage 5 log for backward-compat finalize()."""
    log_output_pipeline(
        log=log,
        req_id=req_id,
        intent=intent,
        sql="<from_result>",
        duration_ms=duration_ms,
        status=status,
        l5_detail=l5_details or "",
        l6_detail=l6_details or "",
    )


def log_final_answer(log, answer: str, intent: str, sql: Optional[str] = None):
    """Legacy shim — no-op (answer is already formatted by formatter)."""
    pass
