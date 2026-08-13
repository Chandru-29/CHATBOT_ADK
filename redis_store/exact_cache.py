"""
exact_cache.py — Redis Exact Query Cache for WMS AI Chatbot.

Caches exact question + role responses in Redis using key pattern:
    sql_chatbot:exact:{role}:{md5_hash}
"""

# ── MODULE TAG: Redis Exact Response Cache ──
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any

from redis_store.client import redis_manager
from core.config.logger import get_logger

log = get_logger(__name__)


def _compute_cache_key(question: str, role: str = "user") -> str:
    """Generate MD5 hash Redis key string for question + role combination.

    Args:
        question (str): User question string.
        role (str, optional): User role string. Defaults to "user".

    Returns:
        str: MD5-hashed Redis key string.
    """
    raw_str = f"{role.strip().lower()}:{question.strip().lower()}"
    hash_digest = hashlib.md5(raw_str.encode("utf-8")).hexdigest()
    return f"sql_chatbot:exact:{role.strip().lower()}:{hash_digest}"


class RedisExactCache:
    """Redis-backed exact text query result cache."""

    def __init__(self) -> None:
        """Initialize RedisExactCache instance."""
        pass

    async def get_async(self, question: str, role: str = "user") -> Optional[Dict[str, Any]]:
        """Asynchronously check Redis for an exact matching query result.

        Args:
            question (str): User question string.
            role (str, optional): User role string. Defaults to "user".

        Returns:
            Optional[Dict[str, Any]]: Parsed result dictionary if hit, or None if miss/offline.
        """
        client = redis_manager.get_client()
        if client is None:
            return None

        try:
            key = _compute_cache_key(question, role)
            val = await client.get(key)
            if val:
                log.info(f"RedisExactCache: HIT for key '{key}'")
                return json.loads(val)
        except Exception as e:
            log.warning(f"RedisExactCache: get_async failed ({e}) — falling back to miss.")
        return None

    def get(self, question: str, role: str = "user") -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for get_async.

        Args:
            question (str): User question string.
            role (str, optional): User role string. Defaults to "user".

        Returns:
            Optional[Dict[str, Any]]: Parsed result dictionary if hit, or None if miss/offline.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self.get_async(question, role), loop)
                return future.result(timeout=2.0)
            else:
                return loop.run_until_complete(self.get_async(question, role))
        except Exception:
            return None

    async def set_async(
        self,
        question: str,
        role: str = "user",
        sql_query: Optional[str] = None,
        answer: Optional[str] = None,
        result_dict: Optional[Dict[str, Any]] = None,
        ttl: int = 1800,
    ) -> bool:
        """Asynchronously store exact query result as JSON in Redis with TTL.

        Args:
            question (str): User question string.
            role (str, optional): User role string. Defaults to "user".
            sql_query (Optional[str], optional): Executed SQL statement. Defaults to None.
            answer (Optional[str], optional): Natural language answer. Defaults to None.
            result_dict (Optional[Dict[str, Any]], optional): Complete result dictionary. Defaults to None.
            ttl (int, optional): Expiration TTL in seconds. Defaults to 1800.

        Returns:
            bool: True if key stored successfully, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None:
            return False

        try:
            key = _compute_cache_key(question, role)
            payload = result_dict or {
                "sql": sql_query,
                "question": question,
                "role": role,
            }
            json_data = json.dumps(payload)
            await client.setex(key, ttl, json_data)
            log.debug(f"RedisExactCache: STORED key '{key}' with TTL {ttl}s.")
            return True
        except Exception as e:
            log.warning(f"RedisExactCache: set_async failed ({e}) — skipped.")
            return False

    def set(
        self,
        question: str,
        role: str = "user",
        sql_query: Optional[str] = None,
        answer: Optional[str] = None,
        result_dict: Optional[Dict[str, Any]] = None,
        ttl: int = 1800,
    ) -> bool:
        """Synchronous wrapper for set_async.

        Args:
            question (str): User question string.
            role (str, optional): User role string. Defaults to "user".
            sql_query (Optional[str], optional): Executed SQL statement. Defaults to None.
            answer (Optional[str], optional): Natural language answer. Defaults to None.
            result_dict (Optional[Dict[str, Any]], optional): Complete result dict. Defaults to None.
            ttl (int, optional): Expiration TTL in seconds. Defaults to 1800.

        Returns:
            bool: True if key stored successfully, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.set_async(question, role, sql_query, answer, result_dict, ttl),
                    loop
                )
                return future.result(timeout=2.0)
            else:
                return loop.run_until_complete(
                    self.set_async(question, role, sql_query, answer, result_dict, ttl)
                )
        except Exception:
            return False

    async def delete_async(self, question: str, role: str = "user") -> bool:
        """Evict entry matching question + role from Redis.

        Args:
            question (str): User question string.
            role (str, optional): User role string. Defaults to "user".

        Returns:
            bool: True if evicted, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None:
            return False
        try:
            key = _compute_cache_key(question, role)
            await client.delete(key)
            log.info(f"RedisExactCache: Evicted key '{key}'")
            return True
        except Exception as e:
            log.warning(f"RedisExactCache: delete_async failed: {e}")
            return False


# Singleton export
redis_exact_cache = RedisExactCache()
