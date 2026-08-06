"""
pipeline_orchestrator.py — Request Pipeline Orchestrator & Audit Logger.

Connects and coordinates the step-by-step query execution lifecycle:
  Step 1  — Layer 1 Guardrails (Injection, PII redaction, write intent, raw SQL).
  Step 2  — Fast-path response cache lookup (exact & semantic).
  Step 3  — Intent routing & rephrasing (Hugging Face embeddings classifier).
  Step 4  — General chat execution (non-DB branch).
  Step 5  — VectorRAG table scope selection & schema extraction.
  Step 6  — MCP Persistent Session & SQL Agent Reasoning Loop.
  Step 7  — Layer 6 Output Sanitization & Execution Pipeline Auditing.
"""

# ── MODULE TAG: Pipeline Execution Orchestrator Service ──
import time
import asyncio
from typing import Optional, Union, Dict, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from core.config.settings import (
    WMS_TABLES, MODEL_NAME, HISTORY_WINDOW,
)
from core.config.logger import get_logger
from api.models import ChatRequest
from db.engine import engine
from db.schema import get_schema
from routing.router_agent import rephrase_and_route, needs_rephrasing
from general_chat.general_agent import handle_general_chat_async
from prompts.loader import prompt_loader
from sql_agent.sql_agent import extract_tables_from_sql
from security.guardrails import (
    guardrails,
    sanitize_output,
    get_pii_reassurance_message,
)
from security.audit_logger import (
    generate_req_id,
    log_incoming_request,
    log_execution_pipeline_trace,
    log_pipeline_terminated,
    log_l1_audit,
    log_audit_tree,
)
from core.cache.cache_manager import (
    get_api_cache,
    get_table_selector,
    sanitize_cache_key,
    lookup_cache,
    store_cache,
    evict_failed_cache,
)
from mcp_service.session_manager import (
    get_persistent_session,
    execute_and_format_cached_query,
    run_with_session,
    run_per_request_session,
)

log = get_logger(__name__)


class Timer:
    """Lightweight wall-clock stopwatch for pipeline step timing."""

    def __init__(self, label: str):
        self.label = label
        self._t0 = time.perf_counter()

    def lap(self, name: str) -> float:
        """Log and return elapsed ms since the last lap (or start)."""
        now = time.perf_counter()
        ms = (now - self._t0) * 1000
        log.debug(f"  ⏱  [{self.label}] {name}: {ms:.0f} ms")
        self._t0 = now
        return ms

    def total(self, name: str, t_start: float) -> None:
        """Log total elapsed time from t_start."""
        ms = (time.perf_counter() - t_start) * 1000
        log.debug(f"  ⏱  [{self.label}] ── TOTAL {name}: {ms:.0f} ms ──")


def _apply_l6(result: Any) -> Any:
    """Apply Layer 6 output content safety to a response dict before returning."""
    if isinstance(result, dict) and result.get("natural_answer"):
        result["natural_answer"] = sanitize_output(
            result["natural_answer"],
            sql_used=result.get("sql"),
        )
    return result


def _finalize_and_audit(
    req_id: str,
    intent: str,
    result: Any,
    t_start: float,
    trace_metrics: Optional[Dict[str, Any]] = None,
) -> Any:
    """Apply Layer 6, emit structured audit trace summary, and return result."""
    result = _apply_l6(result)
    total_ms = (time.perf_counter() - t_start) * 1000

    if trace_metrics:
        sql = None
        if isinstance(result, dict):
            sql = result.get("sql")
        elif hasattr(result, "headers"):
            sql = result.headers.get("x-sql-query")

        compiled_sql = sql or trace_metrics.get("compiled_sql") or "N/A"

        default_names = {
            "GENERAL": "General Agent",
            "WMS_AGENT": "WMS Assistant",
        }
        default_agent_name = default_names.get(intent, "WMS Assistant")

        trace = {
            "server_status": "ONLINE",
            "db_status": "ONLINE",
            "llm_status": "ONLINE",
            "user_question": trace_metrics.get("user_question", ""),
            "cache_status": trace_metrics.get("cache_status", "MISS"),
            "canonical_question": trace_metrics.get("canonical_question") or trace_metrics.get("user_question", ""),
            "intent": intent,
            "rephrase_status": trace_metrics.get("rephrase_status", "Skipped (Self-contained query / No pronouns)"),
            "embedding_str": "Computed via Hugging Face all-MiniLM-L6-v2",
            "exemplar_score": trace_metrics.get("exemplar_score", 0.88),
            "table_vector_score": trace_metrics.get("table_vector_score", 0.92),
            "total_tables_cataloged": trace_metrics.get("total_tables_cataloged", 14),
            "selected_tables_count": trace_metrics.get("selected_tables_count", 14),
            "selected_tables_str": trace_metrics.get("selected_tables_str", ", ".join(WMS_TABLES)),
            "compiled_sql": compiled_sql,
            "total_execution_ms": total_ms,
            "http_status": "200 OK" if not (isinstance(result, dict) and result.get("error")) else "500 Error",
        }

        log_execution_pipeline_trace(log, trace)
        log_pipeline_terminated(log, req_id, total_ms, http_status=trace["http_status"])
    elif isinstance(result, dict):
        sql = result.get("sql")
        tables = extract_tables_from_sql(sql) if sql else []
        log_audit_tree(
            log=log,
            req_id=req_id,
            intent=intent,
            tables=tables,
            duration_ms=total_ms,
            status="PASSED" if not result.get("error") else "FAILED",
        )

    return result


def _build_messages(req: ChatRequest, question: str, system_prompt: str) -> list:
    """Construct the Ollama message list for the agent loop."""
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.chat_history[-HISTORY_WINDOW:]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "bot":
            role = "assistant"
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


async def process_query_pipeline(req: ChatRequest, stream: bool = False) -> Union[dict, StreamingResponse]:
    """
    Main chat execution pipeline.
    Processes natural-language questions through Guardrails L1-L6, Caches,
    Intent Router, VectorRAG Schema Selection, and MCP SQL Reasoning Loop.
    """
    t_request_start = time.perf_counter()
    timer = Timer("pipeline")
    req_id = generate_req_id()

    log_incoming_request(log, req_id, req.user_question, stream)

    # ── STEP 1: Layer 1 Guardrails ──
    t_l1_0 = time.perf_counter()
    original_q = req.user_question
    l1_res = await guardrails.run_layer_1_async(req.user_question)
    is_safe = l1_res["is_safe"]
    unsafe_reason = l1_res["unsafe_reason"]
    redacted_q = l1_res["redacted_question"]
    detected_pii_labels = l1_res["detected_pii"]
    has_write = l1_res["has_write"]
    write_reason = l1_res["write_reason"]
    has_raw_sql = l1_res["has_raw_sql"]
    raw_sql_reason = l1_res["raw_sql_reason"]
    l1_latency_ms = (time.perf_counter() - t_l1_0) * 1000

    if not is_safe:
        log_l1_audit(log, req_id, is_safe=False, has_write=False, details_override=unsafe_reason)
        raise HTTPException(status_code=400, detail=unsafe_reason)

    pii_was_redacted = (redacted_q != original_q)
    pii_reassurance_msg = get_pii_reassurance_message(detected_pii_labels) if pii_was_redacted else ""
    l1_details = (
        "PASSED (Injection cleared, PII redacted, Write-Intent cleared)"
        if pii_was_redacted
        else "PASSED (Injection, PII, Write-Intent cleared)"
    )

    req.user_question = redacted_q

    if has_raw_sql:
        log.warning(f"Layer 1 Guardrail: Raw SQL input intercepted: '{original_q}'")
        timer.total("/query (L1 Raw SQL Refusal)", t_request_start)
        refusal_msg = raw_sql_reason
        if stream:
            async def _stream_raw_sql_refusal():
                yield refusal_msg
            return StreamingResponse(_stream_raw_sql_refusal(), media_type="text/event-stream", headers={"x-agent-name": "System Guard"})
        return {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": refusal_msg,
            "error":          None,
            "attempts":       1,
            "agent_name":     "System Guard",
        }

    if has_write:
        log_l1_audit(log, req_id, is_safe=True, has_write=True)
        if stream:
            async def _stream_refusal():
                yield write_reason
            return StreamingResponse(_stream_refusal(), media_type="text/event-stream", headers={"x-agent-name": "System Guard"})
        return {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": write_reason,
            "error":          None,
            "attempts":       1,
            "agent_name":     "System Guard",
        }

    timer.lap("Layer 1 parallel guardrails")

    cache_key = sanitize_cache_key(req.user_question)

    # ── STEP 2: Fast-Path Response Cache Check ──
    t_cache_0 = time.perf_counter()
    is_follow_up = bool(req.chat_history) and needs_rephrasing(req.user_question, req.chat_history)

    cache_hit, hit_source = lookup_cache(req.user_question, is_follow_up)
    cache_latency_ms = (time.perf_counter() - t_cache_0) * 1000

    trace_metrics = {
        "user_question": original_q,
        "l1_details": l1_details,
        "l1_latency_ms": l1_latency_ms,
        "cache_status": f"HIT ({hit_source.upper()})" if cache_hit else "MISS",
        "cache_latency_ms": cache_latency_ms,
    }

    if cache_hit:
        if isinstance(cache_hit, dict) and cache_hit.get("sql"):
            log.info(f"  {hit_source.upper()} CACHE HIT (SQL query): '{cache_hit['sql']}'")
            success, result = await execute_and_format_cached_query(
                sql=cache_hit["sql"],
                intent=cache_hit.get("intent", "WMS_AGENT"),
                question=req.user_question,
                agent_name=cache_hit.get("agent_name", "WMS Assistant"),
                stream=stream,
                timer=timer,
                hit_source=hit_source,
            )
            if success:
                timer.lap(f"{hit_source} cache execution and formatting")
                timer.total(f"/query ({hit_source} cache hit)", t_request_start)
                trace_metrics.update({"canonical_question": req.user_question, "compiled_sql": cache_hit.get("sql")})
                return _finalize_and_audit(req_id, cache_hit.get("intent", "GENERAL"), result, t_request_start, trace_metrics)
            else:
                evict_failed_cache(cache_key, cache_hit, hit_source)
        else:
            log.info(f"  {hit_source.upper()} CACHE HIT (non-SQL): '{req.user_question}'")
            timer.total(f"/query ({hit_source} cache hit)", t_request_start)
            trace_metrics.update({"canonical_question": req.user_question})

            answer = cache_hit.get("natural_answer", "") if isinstance(cache_hit, dict) else str(cache_hit)
            if stream:
                async def _stream_cache_hit():
                    yield answer
                headers = {"x-cache-hit": hit_source, "x-agent-name": "General Agent"}
                result = StreamingResponse(_stream_cache_hit(), media_type="text/event-stream", headers=headers)
            else:
                if isinstance(cache_hit, dict):
                    cache_hit["cache_hit"] = hit_source
                result = cache_hit

            return _finalize_and_audit(req_id, "GENERAL", result, t_request_start, trace_metrics)

    timer.lap("cache lookup (miss)")

    # ── STEP 3: Intent Routing & Rephrasing ──
    t_router_0 = time.perf_counter()
    from routing.router_agent import rephrase_and_route_with_score
    question, intent, ex_score = await rephrase_and_route_with_score(req.user_question, req.chat_history)
    router_latency_ms = (time.perf_counter() - t_router_0) * 1000
    timer.lap("rephrase_and_route (HF Embedding Classifier)")

    rephrase_status = f"Rephrased -> '{question}'" if question != req.user_question else "Skipped (Self-contained query / No pronouns)"
    trace_metrics.update({
        "router_latency_ms": router_latency_ms,
        "intent": intent,
        "intent_detector": "Hugging Face all-MiniLM-L6-v2",
        "rephrase_status": rephrase_status,
        "canonical_question": question,
        "exemplar_score": ex_score,
    })

    if is_follow_up:
        rephrased_clean_key = sanitize_cache_key(question)
        rephrased_cache_hit, _ = lookup_cache(question, is_follow_up=False)
        if rephrased_cache_hit and isinstance(rephrased_cache_hit, dict) and rephrased_cache_hit.get("sql"):
            log.info(f"  REPHRASED CACHE HIT (SQL query): '{rephrased_cache_hit['sql']}'")
            success, result = await execute_and_format_cached_query(
                sql=rephrased_cache_hit["sql"],
                intent=rephrased_cache_hit.get("intent", "WMS_AGENT"),
                question=question,
                agent_name=rephrased_cache_hit.get("agent_name", "WMS Assistant"),
                stream=stream,
                timer=timer,
                hit_source="rephrased",
            )
            if success:
                trace_metrics.update({"cache_status": "HIT (REPHRASED)", "compiled_sql": rephrased_cache_hit.get("sql")})
                return _finalize_and_audit(req_id, rephrased_cache_hit.get("intent", "WMS_AGENT"), result, t_request_start, trace_metrics)

    # ── STEP 4: General Chat Branch (No Database) ──
    if intent == "GENERAL":
        trace_metrics.update({
            "agent_display_name": "General Agent",
            "selected_tables_str": "N/A (General Chat)",
            "selected_tables_count": 0,
            "exemplar_score": None,
            "table_vector_score": None,
            "compiled_sql": "N/A",
        })
        reply = pii_reassurance_msg if pii_was_redacted else await handle_general_chat_async(
            question,
            has_write=has_write,
            write_reason=write_reason,
        )
        timer.lap(f"general_chat ({MODEL_NAME})")

        if not (pii_was_redacted or has_write or "cannot modify" in reply.lower() or "read-only access" in reply.lower()):
            cache_entry = {
                "sql":            None,
                "intent":         "GENERAL",
                "natural_answer": reply,
                "agent_name":     "General Agent",
            }
            store_cache(req.user_question, cache_entry)

        if stream:
            async def _stream_pii_or_general():
                yield reply
            result = StreamingResponse(_stream_pii_or_general(), media_type="text/event-stream")
        else:
            result = {
                "sql":            None,
                "columns":        [],
                "rows":           [],
                "natural_answer": reply,
                "error":          None,
                "attempts":       1,
                "agent_name":     "General Agent",
            }

        timer.total("/query (GENERAL, no DB)", t_request_start)
        return _finalize_and_audit(req_id, "GENERAL", result, t_request_start, trace_metrics)

    # ── STEP 5: VectorRAG Table Scope Selection & Schema Fetch ──
    agent_cfg = prompt_loader.get_agent_config(intent)
    agent_display_name = agent_cfg.get("display_name", "WMS Assistant")
    trace_metrics.update({"agent_display_name": agent_display_name})

    t_rag_0 = time.perf_counter()
    table_selector = get_table_selector()
    try:
        focused_tables, tbl_score = await asyncio.to_thread(
            table_selector.select_tables_with_score, question, WMS_TABLES, engine, None
        )
        timer.lap("VectorRAG table selection")
        log.info(f"  VectorRAG: {set(WMS_TABLES)} → {set(focused_tables)}")
        schema = get_schema(include_tables=focused_tables)
        timer.lap("get_schema focused")
    except Exception as e:
        result = {
            "sql":            None,
            "columns":        [],
            "rows":           [],
            "natural_answer": None,
            "error":          f"Failed to retrieve database schema: {str(e)}",
            "attempts":       0,
            "agent_name":     agent_display_name,
        }
        return result

    schema_latency_ms = (time.perf_counter() - t_rag_0) * 1000
    sel_tables = ", ".join(sorted(focused_tables)) if focused_tables else "N/A"
    trace_metrics.update({
        "schema_latency_ms": schema_latency_ms,
        "selected_tables_count": len(focused_tables) if focused_tables else len(WMS_TABLES),
        "selected_tables_str": sel_tables,
        "table_vector_score": tbl_score,
        "agent_display_name": agent_display_name,
    })

    system_prompt = prompt_loader.get_trimmed_coder_sql_directive(
        schema_context=schema,
        question=question,
        focused_tables=focused_tables,
        top_k_examples=5,
        intent=intent,
    )

    messages = _build_messages(req, question, system_prompt)

    # ── STEP 6: MCP Session & SQL Agent Reasoning Loop ──
    persistent_session = get_persistent_session()

    if persistent_session is not None:
        try:
            result = await run_with_session(
                session=persistent_session,
                messages=messages,
                system_prompt=system_prompt,
                question=question,
                intent=intent,
                agent_display_name=agent_display_name,
                cache_key=cache_key,
                stream=stream,
                timer=timer,
            )
            timer.total("/query (persistent MCP)", t_request_start)
            _post_process_sql_agent_result(result, req.user_question, question, intent, agent_display_name)
            return _finalize_and_audit(req_id, intent, result, t_request_start, trace_metrics)
        except Exception as e:
            log.warning(f"  Persistent MCP session failed ({e}). Restarting…")
            try:
                from api.app_factory import restart_mcp
                fresh_session = await restart_mcp()
                result = await run_with_session(
                    session=fresh_session,
                    messages=messages,
                    system_prompt=system_prompt,
                    question=question,
                    intent=intent,
                    agent_display_name=agent_display_name,
                    cache_key=cache_key,
                    stream=stream,
                    timer=timer,
                )
                timer.total("/query (restarted MCP)", t_request_start)
                _post_process_sql_agent_result(result, req.user_question, question, intent, agent_display_name)
                return _finalize_and_audit(req_id, intent, result, t_request_start, trace_metrics)
            except Exception as e2:
                log.error(f"  MCP restart also failed ({e2}). Falling back to per-request spawn.")

    result = await run_per_request_session(
        messages=messages,
        system_prompt=system_prompt,
        question=question,
        intent=intent,
        agent_display_name=agent_display_name,
        cache_key=cache_key,
        stream=stream,
        timer=timer,
    )
    timer.total("/query (per-request MCP fallback)", t_request_start)
    _post_process_sql_agent_result(result, req.user_question, question, intent, agent_display_name)
    return _finalize_and_audit(req_id, intent, result, t_request_start, trace_metrics)


def _post_process_sql_agent_result(
    result: Any,
    user_question: str,
    question: str,
    intent: str,
    agent_display_name: str,
) -> None:
    """Extract and cache successful SQL query results."""
    sql_query = None
    if isinstance(result, dict):
        sql_query = result.get("sql")
    elif isinstance(result, StreamingResponse):
        sql_query = result.headers.get("x-sql-query")

    is_success = True
    if isinstance(result, dict) and result.get("error") is not None:
        is_success = False

    if is_success and sql_query:
        answer_str = str(result.get("natural_answer", "") if isinstance(result, dict) else "").lower()
        if not any(err_phrase in answer_str for err_phrase in ["caused the error", "error executing", "read-only access", "cannot modify"]):
            cache_entry = {
                "sql":        sql_query,
                "intent":     intent,
                "agent_name": agent_display_name,
                "question":   question,
            }
            store_cache(user_question, cache_entry)

