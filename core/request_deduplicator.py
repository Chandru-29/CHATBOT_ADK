"""
request_deduplicator.py — In-flight request coalescing for identical concurrent queries.

When multiple users submit the same question simultaneously (common under burst load),
only one pipeline execution is launched. All other concurrent callers await the same
asyncio.Future and receive the result when it resolves — without triggering any
additional LLM calls, DB queries, or MCP session allocations.

This is especially effective for popular questions like "how many items are pending?"
that may arrive from 10-20 users at once.

Usage:
    from core.request_deduplicator import deduplicated_run

    result = await deduplicated_run(
        key=cache_key,
        coro_factory=lambda: _run_pipeline_inner(req, stream=False),
    )

Note:
    Only use with non-streaming requests. Streaming responses produce generators
    that cannot be shared across coroutines.
"""

# ── MODULE TAG: In-Flight Request Deduplicator ──
import asyncio
from typing import Any, Callable, Coroutine

from core.config.logger import get_logger

log = get_logger(__name__)

# Registry of in-flight pipeline executions: cache_key → asyncio.Future
_in_flight: dict[str, asyncio.Future] = {}


async def deduplicated_run(
    key: str,
    coro_factory: Callable[[], Coroutine],
) -> Any:
    """Execute coro_factory() exactly once for the given key, coalescing concurrent calls.

    If an identical key is already in-flight (i.e., another coroutine is executing
    the same pipeline), this coroutine awaits the same Future instead of launching
    a duplicate execution.

    Args:
        key (str): A string key identifying the request (typically the sanitized cache key).
        coro_factory (Callable[[], Coroutine]): A zero-argument callable returning a coroutine.

    Returns:
        Any: Result of the coroutine execution (dict response payload).

    Raises:
        Exception: Any exception raised by the coroutine is re-raised in all waiting callers.
    """
    if key in _in_flight:
        log.info(f"RequestDeduplicator: Coalescing onto in-flight request for key='{key[:60]}'")
        try:
            return await asyncio.shield(_in_flight[key])
        except Exception:
            log.warning(f"RequestDeduplicator: In-flight request for key='{key[:60]}' failed. Retrying.")

    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _in_flight[key] = future

    try:
        result = await coro_factory()
        future.set_result(result)
        return result
    except Exception as exc:
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        _in_flight.pop(key, None)


def get_dedup_stats() -> dict:
    """Retrieve active deduplicator monitoring metrics.

    Returns:
        dict: Dict containing `in_flight_count` and `in_flight_keys` list.
    """
    return {
        "in_flight_count": len(_in_flight),
        "in_flight_keys": list(_in_flight.keys())[:10],
    }
