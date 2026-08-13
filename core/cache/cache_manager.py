"""
cache_manager.py — Centralized Unified Cache Manager with Redis & Memory Fallbacks.

Manages:
  1. Redis Exact Cache (redis_exact_cache) & memory TTLCache (api_cache)
  2. Redis Vector Cache (redis_semantic_cache) & ChromaDB SemanticCache (_sem_cache)
  3. TableSelector (_table_selector) — Vector schema RAG indexer
  4. Unified lookup, store, and cache clearing routines with automatic fallback.
"""

# ── MODULE TAG: Centralized Unified Cache Manager ──
import re
import asyncio
from cachetools import TTLCache
from typing import Optional, Tuple, Dict, Any

from core.config.settings import API_CACHE_TTL, DEFAULT_EMBED_MODEL
from core.config.logger import get_logger
from core.cache.semantic_cache import SemanticCache
from redis_store import redis_manager, redis_exact_cache, redis_semantic_cache
from core.llm.embedder import TextEmbedder
from db.table_selector import TableSelector

log = get_logger(__name__)

# Local in-memory fallback instances
# maxsize raised to 2000 so the cache doesn't evict under 100+ concurrent users
api_cache: dict = TTLCache(maxsize=2000, ttl=API_CACHE_TTL)
_embedder = TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)
_sem_cache = SemanticCache(embedder=_embedder)
_table_selector = TableSelector(use_api_embeddings=True)


def get_api_cache() -> dict:
    """Get the active local memory cache object.

    Returns:
        dict: The active memory cache dictionary.
    """
    return api_cache


def get_table_selector() -> TableSelector:
    """Get the active database table selection tool.

    Returns:
        TableSelector: The active TableSelector object.
    """
    return _table_selector


def sanitize_cache_key(question: str) -> str:
    """Clean and simplify a question into a standard cache lookup key string.

    Args:
        question (str): The user's question text.

    Returns:
        str: Cleaned, lowercase text with no punctuation.
    """
    clean = re.sub(r'[^\w\s]', '', question)
    return clean.strip().lower()


async def async_lookup_cache(user_question: str, is_follow_up: bool, role: str = "user") -> Tuple[Optional[dict], Optional[str]]:
    """Search for a saved answer across all cache systems (Redis and Local Memory).

    Args:
        user_question (str): The question text.
        is_follow_up (bool): True if this is a follow-up question, False if standalone.
        role (str, optional): User role string. Defaults to "user".

    Returns:
        Tuple[Optional[dict], Optional[str]]: A tuple containing the cached answer dict (or None) and the cache type name (`"exact"` or `"semantic"`).
    """
    if is_follow_up:
        return None, None

    cache_key = sanitize_cache_key(user_question)

    # Tier 1: Redis Exact Cache
    if redis_manager.is_connected:
        try:
            exact_hit = await redis_exact_cache.get_async(user_question, role=role)
            if exact_hit:
                log.info(f"  REDIS EXACT CACHE HIT: '{user_question[:60]}'")
                return exact_hit, "exact"
        except Exception as e:
            log.warning(f"  Redis exact cache lookup error ({e}) — trying semantic/memory fallback.")

    # Tier 2: Memory TTLCache Fallback
    if cache_key in api_cache:
        log.info(f"  MEMORY EXACT CACHE HIT: '{user_question[:60]}'")
        return api_cache[cache_key], "exact"

    # Tier 3: Redis Semantic Vector Cache
    if redis_manager.is_connected:
        try:
            sem_hit = await redis_semantic_cache.lookup_async(user_question)
            if sem_hit:
                log.info(f"  REDIS SEMANTIC CACHE HIT: '{user_question[:60]}'")
                return sem_hit, "semantic"
        except Exception as e:
            log.warning(f"  Redis semantic cache lookup error ({e}) — trying ChromaDB fallback.")

    # Tier 4: ChromaDB Semantic Cache Fallback
    chroma_hit = _sem_cache.lookup(user_question)
    if chroma_hit:
        log.info(f"  CHROMADB SEMANTIC CACHE HIT: '{user_question[:60]}'")
        return chroma_hit, "semantic"

    return None, None


async def lookup_cache(user_question: str, is_follow_up: bool, role: str = "user") -> Tuple[Optional[dict], Optional[str]]:
    """Async entry point to search for saved answers in cache.

    Args:
        user_question (str): The user's question text.
        is_follow_up (bool): True if follow-up question.
        role (str, optional): User role string. Defaults to "user".

    Returns:
        Tuple[Optional[dict], Optional[str]]: Cached answer dict and cache source name.
    """
    return await async_lookup_cache(user_question, is_follow_up, role)


async def async_store_cache(user_question: str, entry: dict, role: str = "user") -> None:
    """Save an answer into both Redis and local memory caches.

    Args:
        user_question (str): The user's question text.
        entry (dict): The answer data dictionary to save.
        role (str, optional): User role string. Defaults to "user".
    """
    cache_key = sanitize_cache_key(user_question)
    api_cache[cache_key] = entry

    if redis_manager.is_connected:
        try:
            await redis_exact_cache.set_async(user_question, role=role, result_dict=entry)
            await redis_semantic_cache.store_async(user_question, entry)
        except Exception as e:
            log.warning(f"  Failed to store entry in Redis ({e}). Saved in local memory.")

    _sem_cache.store(user_question, entry)


async def _background_chroma_store(user_question: str, entry: dict) -> None:
    """Save answer to ChromaDB memory in a background thread so the system stays fast.

    Args:
        user_question (str): The user's question text.
        entry (dict): Answer data dictionary.
    """
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_sem_cache.store, user_question, entry),
            timeout=2.0,
        )
    except Exception as e:
        log.debug(f"Background ChromaDB store skipped or timed out: {e}")


def store_cache(user_question: str, entry: dict, role: str = "user") -> None:
    """Save an answer into memory immediately and start background saves to Redis.

    Args:
        user_question (str): The user's question text.
        entry (dict): Answer data dictionary to store.
        role (str, optional): User role string. Defaults to "user".
    """
    cache_key = sanitize_cache_key(user_question)
    api_cache[cache_key] = entry

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(redis_exact_cache.set_async(user_question, role=role, result_dict=entry))
            asyncio.create_task(redis_semantic_cache.store_async(user_question, entry))
            asyncio.create_task(_background_chroma_store(user_question, entry))
        else:
            loop.run_until_complete(async_store_cache(user_question, entry, role=role))
    except Exception:
        pass


def evict_failed_cache(cache_key: str, cache_entry: dict, hit_source: str) -> None:
    """Remove a broken or outdated answer from memory and vector caches.

    Args:
        cache_key (str): Exact cache key string.
        cache_entry (dict): Answer data dictionary to remove.
        hit_source (str): Cache source name (`"exact"` or `"semantic"`).
    """
    log.warning(f"  {hit_source.upper()} CACHE HIT failed execution. Evicting entry.")
    if hit_source == "exact":
        api_cache.pop(cache_key, None)
    else:
        _sem_cache.remove(cache_entry)


def clear_all_caches() -> None:
    """Wipe all saved answers, table structures, security rules, and prompt settings from memory."""
    api_cache.clear()
    _sem_cache.clear()
    _table_selector.clear_cache()

    from db.schema import clear_schema_cache
    clear_schema_cache()

    from security.guardrails import reload_guardrails_config
    reload_guardrails_config()

    from prompts.loader import prompt_loader
    prompt_loader.reload()

    log.info("All unified backend caches, guardrails, and prompt configurations reloaded.")
