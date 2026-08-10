"""
cache_manager.py — Centralized Cache Management for Response & Vector Caches.

Manages:
  1. Exact-text TTLCache (api_cache) — fast secondary cache lookup.
  2. ChromaDB SemanticCache (_sem_cache) — primary vector paraphrase cache.
  3. TableSelector (_table_selector) — Vector schema RAG indexer.
  4. Cache clearing & invalidation routines.
"""

# ── MODULE TAG: Centralized Cache Manager ──
import re
from cachetools import TTLCache
from typing import Optional

from core.config.settings import API_CACHE_TTL, DEFAULT_EMBED_MODEL
from core.config.logger import get_logger
from core.cache.semantic_cache import SemanticCache
from core.llm.embedder import TextEmbedder
from db.table_selector import TableSelector

log = get_logger(__name__)

api_cache: dict = TTLCache(maxsize=100, ttl=API_CACHE_TTL)
_embedder = TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)
_sem_cache = SemanticCache(embedder=_embedder)
_table_selector = TableSelector(use_api_embeddings=True)


def get_api_cache() -> dict:
    """Return active exact-text TTLCache instance."""
    return api_cache


def get_table_selector() -> TableSelector:
    """Return active TableSelector instance."""
    return _table_selector


def sanitize_cache_key(question: str) -> str:
    """Clean question string for exact cache lookup key."""
    clean = re.sub(r'[^\w\s]', '', question)
    return clean.strip().lower()


def lookup_cache(user_question: str, is_follow_up: bool) -> tuple[Optional[dict], Optional[str]]:
    """Perform exact and semantic cache lookup for *user_question*."""
    if is_follow_up:
        return None, None

    cache_key = sanitize_cache_key(user_question)

    if cache_key in api_cache:
        log.info(f"  EXACT CACHE HIT: '{user_question}'")
        return api_cache[cache_key], "exact"

    sem_hit = _sem_cache.lookup(user_question)
    if sem_hit:
        log.info(f"  SEMANTIC CACHE HIT: '{user_question}'")
        return sem_hit, "semantic"

    return None, None


def store_cache(user_question: str, entry: dict) -> None:
    """Store result entry in both exact TTLCache and ChromaDB semantic cache."""
    cache_key = sanitize_cache_key(user_question)
    api_cache[cache_key] = entry
    _sem_cache.store(user_question, entry)


def evict_failed_cache(cache_key: str, cache_entry: dict, hit_source: str) -> None:
    """Evict an invalid/failed query result from cache."""
    log.warning(f"  {hit_source.upper()} CACHE HIT failed execution. Evicting entry.")
    if hit_source == "exact":
        api_cache.pop(cache_key, None)
    else:
        _sem_cache.remove(cache_entry)


def clear_all_caches() -> None:
    """Clear all system caches across database, guardrails, and prompts."""
    api_cache.clear()
    _sem_cache.clear()
    _table_selector.clear_cache()

    from db.schema import clear_schema_cache
    clear_schema_cache()

    from security.guardrails import reload_guardrails_config
    reload_guardrails_config()

    from prompts.loader import prompt_loader
    prompt_loader.reload()

    log.info("All backend caches, guardrails, and prompt configurations reloaded.")
