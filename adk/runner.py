"""
runner.py — ADK Workflow Orchestrator & Runner Engine.

Coordinates multi-agent delegation (RouterAgent -> WMSSQLAgent / GeneralChatAgent),
manages session state, integrates ADK Middleware hooks, and formats responses.
"""

# ── MODULE TAG: ADK Workflow Runner Engine ──
import time
import asyncio
from typing import Union, Dict, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from core.config.settings import WMS_TABLES, MODEL_NAME, HISTORY_WINDOW, VECTOR_RAG_THRESHOLD
from core.config.logger import get_logger
from api.models import ChatRequest
from db.engine import engine
from db.schema import get_schema
from agents.router_agent import rephrase_and_route_with_score, needs_rephrasing
from agents.general_agent import handle_general_chat_async
from agents.sql_agent import extract_tables_from_sql
from prompts.loader import prompt_loader
from security.audit_logger import (
    generate_req_id,
    log_incoming_request,
    log_l1_audit,
    log_l2_cache,
    log_router_decision,
    log_rag_selection,
    log_tool_execution,
    log_l3_l4_guardrails,
    log_output_pipeline,
    # legacy shims kept for any other callers
    log_execution_pipeline_trace,
    log_pipeline_terminated,
    log_audit_tree,
)
from core.cache.cache_manager import (
    get_table_selector,
    sanitize_cache_key,
    store_cache,
    evict_failed_cache,
)
from mcp_service.session_manager import (
    get_persistent_session,
    execute_and_format_cached_query,
    run_with_session,
    run_per_request_session,
)
from adk.middleware import ADKMiddleware

log = get_logger(__name__)


class ADKRunner:
    """
    Main ADK Pipeline Workflow Orchestrator.
    Manages end-to-end multi-agent execution lifecycle.
    """

    @staticmethod
    async def run_pipeline(req: ChatRequest, stream: bool = False) -> Union[dict, StreamingResponse]:
        """
        Execute ADK Multi-Agent Workflow Pipeline.
        """
        t_request_start = time.perf_counter()
        req_id = generate_req_id()

        # ── [1] CLIENT & RUNNER ENTRY ──────────────────────────────────────
        log_incoming_request(log, req_id, req.user_question, stream)

        # ── [2a] MIDDLEWARE: Layer 1 Input Guardrails ──────────────────────
        original_q = req.user_question
        is_pass, clean_q_or_refusal, l1_meta = await ADKMiddleware.process_l1_input(original_q)

        log_l1_audit(log, req_id, is_safe=is_pass, has_write=l1_meta.get("has_write", False))

        if not is_pass:
            if stream:
                async def _stream_refusal():
                    yield clean_q_or_refusal
                return StreamingResponse(
                    _stream_refusal(),
                    media_type="text/event-stream",
                    headers={"x-agent-name": "System Guard"},
                )
            return {
                "sql": None,
                "columns": [],
                "rows": [],
                "natural_answer": clean_q_or_refusal,
                "error": None,
                "attempts": 1,
                "agent_name": "System Guard",
            }

        req.user_question = clean_q_or_refusal
        cache_key = sanitize_cache_key(req.user_question)

        # ── [2b] MIDDLEWARE: Layer 2 Fast-Path Cache Check ─────────────────
        is_follow_up = bool(req.chat_history) and needs_rephrasing(req.user_question, req.chat_history)
        cache_hit, cache_payload, hit_source = ADKMiddleware.check_l2_cache(req.user_question, is_follow_up)

        log_l2_cache(log, hit=cache_hit, hit_source=hit_source)

        if cache_hit:
            if isinstance(cache_payload, dict) and cache_payload.get("sql"):
                log.info(f"  └── Cache SQL: {cache_payload['sql']}")
                success, result = await execute_and_format_cached_query(
                    sql=cache_payload["sql"],
                    intent=cache_payload.get("intent", "WMS_AGENT"),
                    question=req.user_question,
                    agent_name=cache_payload.get("agent_name", "WMS Assistant"),
                    stream=stream,
                    hit_source=hit_source,
                )
                if success:
                    return ADKRunner._finalize(req_id, cache_payload.get("intent", "WMS_AGENT"), result, t_request_start)
                else:
                    evict_failed_cache(cache_key, cache_payload, hit_source)
            else:
                answer = cache_payload.get("natural_answer", "") if isinstance(cache_payload, dict) else str(cache_payload)
                if stream:
                    async def _stream_cache():
                        yield answer
                    return StreamingResponse(_stream_cache(), media_type="text/event-stream", headers={"x-agent-name": "General Agent"})
                return ADKRunner._finalize(req_id, "GENERAL", cache_payload, t_request_start)

        # ── [3] ROUTER AGENT: Intent Classification & Rephrasing ──────────
        question, intent, ex_score = await rephrase_and_route_with_score(req.user_question, req.chat_history)
        was_rephrased = question.strip().lower() != req.user_question.strip().lower()
        log_router_decision(
            log,
            original_q=req.user_question,
            rephrased_q=question,
            intent=intent,
            confidence=ex_score,
            was_rephrased=was_rephrased,
        )

        # ── [4a] GENERAL AGENT Branch (No DB) ─────────────────────────────
        if intent == "GENERAL":
            reply = l1_meta["pii_msg"] if l1_meta["pii_redacted"] else await handle_general_chat_async(
                question,
                has_write=l1_meta.get("has_write", False),
                write_reason="",
            )
            if not l1_meta["pii_redacted"]:
                store_cache(req.user_question, {"sql": None, "intent": "GENERAL", "natural_answer": reply, "agent_name": "General Agent"})

            if stream:
                async def _stream_gen():
                    yield reply
                result = StreamingResponse(_stream_gen(), media_type="text/event-stream")
            else:
                result = {
                    "sql": None,
                    "columns": [],
                    "rows": [],
                    "natural_answer": reply,
                    "error": None,
                    "attempts": 1,
                    "agent_name": "General Agent",
                }
            return ADKRunner._finalize(req_id, "GENERAL", result, t_request_start)

        # ── [4b] VectorRAG Schema Scope & Schema Context ───────────────────
        agent_cfg = prompt_loader.get_agent_config(intent)
        agent_display_name = agent_cfg.get("display_name", "WMS Assistant")
        table_selector = get_table_selector()

        try:
            focused_tables, tbl_score = await asyncio.to_thread(
                table_selector.select_tables_with_score, question, WMS_TABLES, engine, None
            )
            schema = get_schema(include_tables=focused_tables)
        except Exception as e:
            return {
                "sql": None,
                "columns": [],
                "rows": [],
                "natural_answer": None,
                "error": f"Failed to retrieve schema: {e}",
                "attempts": 0,
                "agent_name": agent_display_name,
            }

        log_rag_selection(log, tables=list(focused_tables), sim_score=tbl_score, threshold=VECTOR_RAG_THRESHOLD)

        system_prompt = prompt_loader.get_trimmed_coder_sql_directive(
            schema_context=schema,
            question=question,
            focused_tables=focused_tables,
            top_k_examples=5,
            intent=intent,
        )

        messages = ADKRunner._build_messages(req, question, system_prompt)

        # ── [4c] WMSSQLAgent Reasoning Loop ───────────────────────────────
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
                )
                ADKRunner._post_process(result, req.user_question, question, intent, agent_display_name)
                return ADKRunner._finalize(req_id, intent, result, t_request_start)
            except Exception as e:
                log.warning(f"  └── ⚠️ MCP persistent session error: {e}. Falling back to per-request session.")

        result = await run_per_request_session(
            messages=messages,
            system_prompt=system_prompt,
            question=question,
            intent=intent,
            agent_display_name=agent_display_name,
            cache_key=cache_key,
            stream=stream,
        )
        ADKRunner._post_process(result, req.user_question, question, intent, agent_display_name)
        return ADKRunner._finalize(req_id, intent, result, t_request_start)

    @staticmethod
    def _build_messages(req: ChatRequest, question: str, system_prompt: str) -> list:
        """Construct chat message payload."""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in req.chat_history[-HISTORY_WINDOW:]:
            role = msg.get("role", "user")
            if role in ("bot", "assistant"):
                role = "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _post_process(result: Any, user_question: str, question: str, intent: str, agent_display_name: str) -> None:
        """Cache successful SQL execution results."""
        if isinstance(result, dict) and not result.get("error") and result.get("sql"):
            store_cache(user_question, {
                "sql": result.get("sql"),
                "intent": intent,
                "agent_name": agent_display_name,
                "question": question,
            })

    @staticmethod
    def _finalize(req_id: str, intent: str, result: Any, t_start: float) -> Any:
        """Apply ADK L6 Output Sanitizer and emit Stage 5 audit log."""
        result = ADKMiddleware.sanitize_l6_output(result)
        total_ms = (time.perf_counter() - t_start) * 1000
        sql = result.get("sql") if isinstance(result, dict) else None
        has_error = isinstance(result, dict) and bool(result.get("error"))
        status = "FAILED" if has_error else "PASSED"

        if sql:
            log_tool_execution(log, sql)
            log_l3_l4_guardrails(log, l3_ok=not has_error, l4_ok=not has_error)

        log_output_pipeline(
            log=log,
            req_id=req_id,
            intent=intent,
            sql=sql,
            duration_ms=total_ms,
            status=status,
        )
        return result
