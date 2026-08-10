"""
middleware.py — ADK Lifecycle Hooks & Security Middleware.

Hooks Layer 1-6 Guardrails, PII Redaction, SQL Validation, Cache Interceptors,
and Execution Pipeline Auditing into ADK Agent lifecycle events.
"""

# ── MODULE TAG: ADK Middleware Security Engine ──
import time
from typing import Dict, Any, Tuple, Optional
from fastapi import HTTPException

from security.guardrails import (
    guardrails,
    validate_sql_before_execution,
    redact_db_output_string,
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
    lookup_cache,
    store_cache,
    sanitize_cache_key,
    evict_failed_cache,
)
from core.config.logger import get_logger

log = get_logger(__name__)


class ADKMiddleware:
    """
    Standardized ADK Security & Auditing Middleware Manager.
    """

    @staticmethod
    async def process_l1_input(question: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Layer 1 Input Guardrail Hook.
        Checks for prompt injection, raw SQL input, write intents, and redacts PII.
        Returns: (is_pass, cleaned_question_or_refusal, l1_metadata)
        """
        t0 = time.perf_counter()
        l1_res = await guardrails.run_layer_1_async(question)
        l1_latency = (time.perf_counter() - t0) * 1000

        is_safe = l1_res["is_safe"]
        unsafe_reason = l1_res["unsafe_reason"]
        redacted_q = l1_res["redacted_question"]
        detected_pii = l1_res["detected_pii"]
        has_write = l1_res["has_write"]
        write_reason = l1_res["write_reason"]
        has_raw_sql = l1_res["has_raw_sql"]
        raw_sql_reason = l1_res["raw_sql_reason"]

        metadata = {
            "l1_latency_ms": l1_latency,
            "detected_pii": detected_pii,
            "has_write": has_write,
            "has_raw_sql": has_raw_sql,
            "pii_redacted": (redacted_q != question),
            "pii_msg": get_pii_reassurance_message(detected_pii) if (redacted_q != question) else "",
        }

        if not is_safe:
            raise HTTPException(status_code=400, detail=unsafe_reason)

        if has_raw_sql:
            return False, raw_sql_reason, metadata

        if has_write:
            return False, write_reason, metadata

        return True, redacted_q, metadata

    @staticmethod
    def check_l2_cache(question: str, is_follow_up: bool) -> Tuple[bool, Optional[Any], str]:
        """
        Layer 2 Cache Interceptor Hook.
        Checks response and SQL cache for exact/semantic hit.
        Returns: (is_hit, cached_payload, hit_source)
        """
        hit, source = lookup_cache(question, is_follow_up)
        return bool(hit), hit, source

    @staticmethod
    def validate_l3_l4_sql(sql_query: str, intent: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Layer 3/4 Pre-Execution SQL Hook.
        Enforces table scope and blocks write queries (INSERT, UPDATE, DELETE, DROP, ALTER).
        Returns: (is_valid, error_message)
        """
        return validate_sql_before_execution(sql_query, intent)

    @staticmethod
    def redact_l5_output(db_output: str) -> str:
        """
        Layer 5 DB Output Hook.
        Redacts sensitive columns (passwords, tokens, PII) from SQL execution outputs.
        """
        return redact_db_output_string(db_output)

    @staticmethod
    def sanitize_l6_output(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Layer 6 Response Sanitizer Hook.
        Sanitizes natural language answer for LLM hallucinations or unauthorized data leaks.
        """
        if isinstance(result, dict) and result.get("natural_answer"):
            result["natural_answer"] = sanitize_output(
                result["natural_answer"],
                sql_used=result.get("sql"),
            )
        return result
