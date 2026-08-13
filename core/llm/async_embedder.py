"""
async_embedder.py — Async text embedding with in-flight deduplication and local model execution.

Key features:
  - In-flight deduplication: if 20 users send the same question simultaneously, only 1
    embedding calculation is executed; the other 19 await the same asyncio.Future.
  - Process-local LRU cache: recently embedded texts are served instantly.
  - Local execution: uses TextEmbedder (SentenceTransformer all-MiniLM-L6-v2) 100% offline.
"""

# ── MODULE TAG: Async Text Embedder ──
import asyncio
from typing import Optional

from core.config.logger import get_logger
from core.llm.embedder import TextEmbedder

log = get_logger(__name__)

# ── Process-local embedding cache ──────────────────────────────────────────────
_embed_cache: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 2048

# ── In-flight deduplication registry ───────────────────────────────────────────
_in_flight: dict[str, asyncio.Future] = {}
_shared_embedder = TextEmbedder()


def _normalize_key(text: str) -> str:
    """Clean and shorten text to use as a unique lookup key.

    Args:
        text (str): The input text to clean.

    Returns:
        str: A clean, lowercase string key.
    """
    return text.strip().lower()[:300]


async def embed_async(text: str) -> list[float]:
    """Convert text to numbers asynchronously without repeating work for identical questions.

    Args:
        text (str): The input text to convert into numbers.

    Returns:
        list[float]: A list of numbers representing the text.
    """
    key = _normalize_key(text)

    # 1. Check process-local cache first (zero latency)
    if key in _embed_cache:
        return _embed_cache[key]

    # 2. If an identical request is already in-flight, await its result
    if key in _in_flight:
        log.debug(f"AsyncEmbedder: Coalescing duplicate embedding request for key='{key[:40]}'")
        try:
            return await asyncio.shield(_in_flight[key])
        except Exception:
            pass

    # 3. Register this request as in-flight
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _in_flight[key] = future

    try:
        vec = await asyncio.to_thread(_shared_embedder.embed, text)
        _store_in_cache(key, vec)
        future.set_result(vec)
        return vec
    except Exception as e:
        log.warning(f"AsyncEmbedder: Embedding failed ({e}). Using mock fallback.")
        vec = _shared_embedder._mock_embed(text)
        _store_in_cache(key, vec)
        if not future.done():
            future.set_result(vec)
        return vec
    finally:
        _in_flight.pop(key, None)


def _store_in_cache(key: str, vec: list[float]) -> None:
    """Save the text numbers in memory cache, removing the oldest entry if full.

    Args:
        key (str): The text lookup key string.
        vec (list[float]): The list of numbers representing text meaning.
    """
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        try:
            del _embed_cache[next(iter(_embed_cache))]
        except StopIteration:
            pass
    _embed_cache[key] = vec


def clear_embed_cache() -> None:
    """Wipe all saved embedding calculations from memory."""
    _embed_cache.clear()
    log.info("AsyncEmbedder: Process-local embedding cache cleared.")
