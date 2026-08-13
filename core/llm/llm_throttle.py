"""
llm_throttle.py — Global concurrency semaphore + token-bucket RPM limiter for Gemini LLM API.

Prevents 429 rate-limit errors under high concurrency by:
  1. Capping the number of simultaneous in-flight LLM API calls (GEMINI_MAX_CONCURRENT).
  2. Enforcing a per-minute request rate ceiling using a token-bucket algorithm (GEMINI_RPM_LIMIT).

Usage:
    from core.llm.llm_throttle import acquire_llm_slot, release_llm_slot

    await acquire_llm_slot()        # blocks until a slot is free (or raises on timeout)
    try:
        result = await llm_call()
    finally:
        release_llm_slot()          # always release

Or use the context-manager helper:
    async with llm_slot():
        result = await llm_call()
"""

# ── MODULE TAG: Gemini LLM Rate-Limit Throttle ──
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from core.config.settings import (
    GEMINI_MAX_CONCURRENT,
    GEMINI_RPM_LIMIT,
    GEMINI_THROTTLE_TIMEOUT,
)
from core.config.logger import get_logger

log = get_logger(__name__)

# ── State (process-local; one per UVICORN worker) ───────────────────────────────
_semaphore: Optional[asyncio.Semaphore] = None

# Token-bucket state
_tb_lock: Optional[asyncio.Lock] = None
_tokens: float = float(GEMINI_RPM_LIMIT)
_last_refill: float = 0.0  # will be set on first acquire


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init the global semaphore (must be created inside a running event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(GEMINI_MAX_CONCURRENT)
        log.info(
            f"LLMThrottle: Initialized semaphore with max_concurrent={GEMINI_MAX_CONCURRENT}, "
            f"rpm_limit={GEMINI_RPM_LIMIT}."
        )
    return _semaphore


def _get_tb_lock() -> asyncio.Lock:
    """Lazy-init the token-bucket asyncio.Lock."""
    global _tb_lock
    if _tb_lock is None:
        _tb_lock = asyncio.Lock()
    return _tb_lock


async def _refill_tokens() -> None:
    """Refill tokens proportional to time elapsed since last refill."""
    global _tokens, _last_refill
    now = time.monotonic()
    if _last_refill == 0.0:
        _last_refill = now
    elapsed = now - _last_refill
    refill = elapsed * (GEMINI_RPM_LIMIT / 60.0)
    _tokens = min(float(GEMINI_RPM_LIMIT), _tokens + refill)
    _last_refill = now


async def acquire_llm_slot(timeout: Optional[float] = None) -> None:
    """
    Acquire a Gemini LLM API slot.

    Blocks until:
      - A concurrency slot is free (semaphore), AND
      - A rate-limit token is available (token bucket).

    Raises RuntimeError if the combined wait exceeds `timeout` seconds.
    """
    global _tokens
    effective_timeout = timeout if timeout is not None else GEMINI_THROTTLE_TIMEOUT

    # Step 1: Acquire concurrency semaphore slot
    sem = _get_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"LLMThrottle: All {GEMINI_MAX_CONCURRENT} concurrent LLM slots are busy. "
            f"Request timed out after {effective_timeout}s. Consider increasing GEMINI_MAX_CONCURRENT."
        )

    # Step 2: Consume a token from the token bucket (while holding the semaphore)
    deadline = time.monotonic() + effective_timeout
    lock = _get_tb_lock()

    async with lock:
        await _refill_tokens()
        while _tokens < 1.0:
            # Calculate how long to wait for one token to accrue
            wait_secs = (1.0 - _tokens) / (GEMINI_RPM_LIMIT / 60.0)
            if time.monotonic() + wait_secs > deadline:
                sem.release()
                raise RuntimeError(
                    f"LLMThrottle: RPM ceiling ({GEMINI_RPM_LIMIT} req/min) reached. "
                    f"Request queued too long. Increase GEMINI_RPM_LIMIT or reduce traffic."
                )
            log.debug(f"LLMThrottle: RPM bucket depleted — waiting {wait_secs:.2f}s for refill.")
            await asyncio.sleep(min(wait_secs, 0.25))  # cap sleep to 250ms for responsiveness
            await _refill_tokens()

        _tokens -= 1.0
        remaining = int(_tokens)
        log.debug(f"LLMThrottle: Slot acquired. Tokens remaining: {remaining}/{GEMINI_RPM_LIMIT}.")


def release_llm_slot() -> None:
    """Release the concurrency semaphore slot after an LLM call completes."""
    sem = _get_semaphore()
    try:
        sem.release()
    except ValueError:
        # Semaphore released more times than acquired — should not happen, but guard anyway
        log.warning("LLMThrottle: release_llm_slot() called without a matching acquire.")


@asynccontextmanager
async def llm_slot(timeout: Optional[float] = None):
    """
    Async context manager for acquiring and releasing a Gemini LLM API slot.

    Usage:
        async with llm_slot():
            result = await llm_call()
    """
    await acquire_llm_slot(timeout=timeout)
    try:
        yield
    finally:
        release_llm_slot()


def get_throttle_status() -> dict:
    """Return current throttle state for health-check / monitoring endpoints."""
    sem = _get_semaphore()
    return {
        "max_concurrent": GEMINI_MAX_CONCURRENT,
        "available_slots": sem._value,  # type: ignore[attr-defined]
        "rpm_limit": GEMINI_RPM_LIMIT,
        "tokens_remaining": round(_tokens, 2),
        "throttle_timeout_s": GEMINI_THROTTLE_TIMEOUT,
    }
