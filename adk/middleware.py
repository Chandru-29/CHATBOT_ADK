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
    """Standardized ADK Security & Auditing Middleware Manager.

    Integrates Layer 1-6 StitchGuard security hooks, PII redaction, cache checking,
    and output sanitization into the pipeline lifecycle.
    """

    @staticmethod
    async def process_l1_input(question: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run Layer 1 Input Guardrail checks for prompt injection, raw SQL input, and PII.

        Args:
            question (str): Raw input user question string.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: A tuple containing:
                - bool: True if safe to proceed, False if security rule intercepted.
                - str: Cleaned/redacted question string or security refusal message.
                - Dict[str, Any]: Layer 1 metadata dict containing PII flags and latency metrics.

        Raises:
            HTTPException: If prompt injection or severe malicious payload is detected.
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
    async def check_l2_cache(question: str, is_follow_up: bool) -> Tuple[bool, Optional[Any], str]:
        """Run Layer 2 Cache Interceptor check against local TTLCache and Redis stores.

        Args:
            question (str): Cleaned user question string.
            is_follow_up (bool): True if question is a multi-turn follow-up query.

        Returns:
            Tuple[bool, Optional[Any], str]: A tuple containing:
                - bool: True if cache hit occurred, False on cache miss.
                - Optional[Any]: Cached response payload object or dict.
                - str: Cache hit source label (`"EXACT"`, `"SEMANTIC"`, or `"NONE"`).
        """
        hit, source = await lookup_cache(question, is_follow_up)
        return bool(hit), hit, source

    @staticmethod
    def validate_l3_l4_sql(sql_query: str, intent: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Run Layer 3/4 Pre-Execution SQL validation guardrails.

        Args:
            sql_query (str): Generated SQL SELECT statement to validate.
            intent (Optional[str], optional): Classified domain intent string. Defaults to None.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing:
                - bool: True if query passes all table scope and read-only checks, False otherwise.
                - Optional[str]: Error refusal message if invalid, or None if valid.
        """
        return validate_sql_before_execution(sql_query, intent)

    @staticmethod
    def redact_l5_output(db_output: str) -> str:
        """Run Layer 5 Database Output Redaction hook.

        Args:
            db_output (str): Raw text output string from database query execution.

        Returns:
            str: Output string with sensitive columns (passwords, tokens, PII) redacted.
        """
        return redact_db_output_string(db_output)

    @staticmethod
    def sanitize_l6_output(result: Dict[str, Any]) -> Dict[str, Any]:
        """Run Layer 6 Response Sanitizer hook on natural language answers.

        Args:
            result (Dict[str, Any]): Execution response payload dictionary.

        Returns:
            Dict[str, Any]: Payload dictionary with sanitized `natural_answer` string.
        """
        if isinstance(result, dict) and result.get("natural_answer"):
            result["natural_answer"] = sanitize_output(
                result["natural_answer"],
                sql_used=result.get("sql"),
            )
        return result
