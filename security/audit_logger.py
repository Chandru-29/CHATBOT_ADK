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
    """Generate a unique tracking ID string for a new chat request.

    Returns:
        str: A short 6-character request ID string (e.g. `"req_9d494e"`).
    """
    return f"req_{uuid.uuid4().hex[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Client & Runner Entry
# ─────────────────────────────────────────────────────────────────────────────

def log_incoming_request(log, req_id: str, query: str, stream: bool = False) -> None:
    """Log the arrival of a new user question at Stage 1.

    Args:
        log: The active logger instance.
        req_id (str): The unique request tracking ID.
        query (str): The question typed by the user.
        stream (bool, optional): True if streaming answer, False otherwise. Defaults to False.
    """
    log.info(f"[1. CLIENT & RUNNER ENTRY]")
    log.info(f"  └── Received Query: \"{query}\" | req_id={req_id} | stream={stream}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Middleware & Guardrails (L1 + L2 cache)
# ─────────────────────────────────────────────────────────────────────────────

def log_l1_audit(log, req_id: str, is_safe: bool, has_write: bool,
                 details_override: Optional[str] = None) -> None:
    """Log Stage 2 Layer 1 safety check results (whether the question was allowed or blocked).

    Args:
        log: The active logger instance.
        req_id (str): Unique request tracking ID.
        is_safe (bool): True if the question is safe and appropriate.
        has_write (bool): True if the question attempts database edits.
        details_override (Optional[str], optional): Custom status text. Defaults to None.
    """
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


def log_l2_cache(log, hit: bool, hit_source: str = "") -> None:
    """Log Stage 2 Layer 2 cache check results (hit or miss).

    Args:
        log: The active logger instance.
        hit (bool): True if answer was found in cache, False if a fresh search is needed.
        hit_source (str, optional): Cache type name (`"EXACT"` or `"SEMANTIC"`). Defaults to "".
    """
    if hit:
        log.info(f"  └── L2 Semantic Cache: Cache HIT ({hit_source.upper()}). Returning cached result.")
    else:
        log.info(f"  └── L2 Semantic Cache: Cache miss. Proceeding to Agent workflow.")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Router Agent & Rephrasing
# ─────────────────────────────────────────────────────────────────────────────

def log_router_decision(log, original_q: str, rephrased_q: str, intent: str,
                        confidence: float, was_rephrased: bool) -> None:
    """Log Stage 3 query rewriting and topic classification results.

    Args:
        log: The active logger instance.
        original_q (str): The original question typed by the user.
        rephrased_q (str): The rewritten standalone question.
        intent (str): The classified category name.
        confidence (float): Classification confidence match score.
        was_rephrased (bool): True if the question was rewritten, False otherwise.
    """
    log.info(f"[3. ROUTER AGENT & REPHRASING (Gemini 2.5 Flash)]")
    if was_rephrased:
        log.info(f"  └── Query Rephraser: Follow-up resolved → \"{rephrased_q}\"")
    else:
        log.info(f"  └── Query Rephraser: Self-contained query — no rephrasing needed.")
    log.info(f"  └── Router Decision: Intent classified as {intent} (Confidence: {confidence:.2f}).")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — WMS SQL Agent & Tool Execution (L3 + L4)
# ─────────────────────────────────────────────────────────────────────────────

def log_rag_selection(log, tables: List[str], sim_score: float, threshold: float) -> None:
    """Log Stage 4 vector table selection results.

    Args:
        log: The active logger instance.
        tables (List[str]): List of selected database tables.
        sim_score (float): Similarity match score for selected tables.
        threshold (float): Minimum required similarity cutoff score.
    """
    log.info(f"[4. WMS SQL AGENT & TOOL EXECUTION (Layer 3-4)]")
    tables_str = str(tables) if tables else "all domain tables (fallback)"
    log.info(f"  └── VectorRAG Table Selector: Grounded tables → {tables_str} (Similarity: {sim_score:.2f} > {threshold:.2f})")


def log_tool_execution(log, sql: str) -> None:
    """Log Stage 4 database query execution details.

    Args:
        log: The active logger instance.
        sql (str): The SQL statement being executed.
    """
    sql_preview = " ".join(sql.split())[:200]
    log.info(f"  └── Tool Execution: Invoking ExecuteReadOnlySQL → {sql_preview}")


def log_l3_l4_guardrails(log, l3_ok: bool, l4_ok: bool,
                          l3_detail: str = "", l4_detail: str = "") -> None:
    """Log Stage 4 Layer 3 (table permissions) and Layer 4 (read-only safety) security checks.

    Args:
        log: The active logger instance.
        l3_ok (bool): True if table scope permissions passed.
        l4_ok (bool): True if query read-only safety passed.
        l3_detail (str, optional): Custom table permission detail message. Defaults to "".
        l4_detail (str, optional): Custom read-only detail message. Defaults to "".
    """
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
                        l5_detail: str = "", l6_detail: str = "") -> None:
    """Log Stage 5 output cleaning, safety checks, and response delivery metrics.

    Args:
        log: The active logger instance.
        req_id (str): Unique request tracking ID.
        intent (str): The question category name.
        sql (Optional[str]): The SQL query that was run (if any).
        duration_ms (float): Total processing time in milliseconds.
        status (str, optional): Execution status string (`"PASSED"` or `"FAILED"`). Defaults to "PASSED".
        l5_detail (str, optional): Column redaction detail message. Defaults to "".
        l6_detail (str, optional): Answer content safety detail message. Defaults to "".
    """
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

def log_execution_pipeline_trace(log, trace: dict) -> None:
    """Old helper shim for pipeline trace logging.

    Args:
        log: The active logger instance.
        trace (dict): Pipeline trace dictionary.
    """
    pass


def log_pipeline_terminated(log, req_id: str, total_ms: float, http_status: str = "200 OK") -> None:
    """Old helper shim for pipeline termination logging.

    Args:
        log: The active logger instance.
        req_id (str): Request tracking ID.
        total_ms (float): Processing time in milliseconds.
        http_status (str, optional): HTTP status string. Defaults to "200 OK".
    """
    pass


def log_audit_tree(log, req_id: str, intent: str, tables: List[str],
                   duration_ms: float, status: str = "PASSED",
                   l3_details: Optional[str] = None,
                   l4_details: Optional[str] = None,
                   l5_details: Optional[str] = None,
                   l6_details: Optional[str] = None,
                   pipeline_info: str = "SQL generated & executed successfully") -> None:
    """Old helper shim for audit tree logging.

    Args:
        log: The active logger instance.
        req_id (str): Unique request tracking ID.
        intent (str): Topic category name.
        tables (List[str]): List of grounded table names.
        duration_ms (float): Processing time in milliseconds.
        status (str, optional): Status label string. Defaults to "PASSED".
        l3_details (Optional[str], optional): L3 detail message. Defaults to None.
        l4_details (Optional[str], optional): L4 detail message. Defaults to None.
        l5_details (Optional[str], optional): L5 detail message. Defaults to None.
        l6_details (Optional[str], optional): L6 detail message. Defaults to None.
        pipeline_info (str, optional): Summary message string. Defaults to "SQL generated & executed successfully".
    """
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


def log_final_answer(log, answer: str, intent: str, sql: Optional[str] = None) -> None:
    """Old helper shim for logging the final response.

    Args:
        log: The active logger instance.
        answer (str): The response text.
        intent (str): Topic category name.
        sql (Optional[str], optional): The SQL query that was run. Defaults to None.
    """
    pass
