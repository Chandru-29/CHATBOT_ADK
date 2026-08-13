"""
session_store.py — Redis-backed Chat Session Thread Manager.

Stores conversation history threads using Redis Lists under key pattern:
    sql_chatbot:session:{session_id}
With an automatic 24-hour TTL expiration.
"""

# ── MODULE TAG: Redis Chat Session Store ──
import json
from typing import List, Dict, Any, Optional

from redis_store.client import redis_manager
from core.config.settings import HISTORY_WINDOW
from core.config.logger import get_logger

log = get_logger(__name__)

SESSION_TTL = 86400  # 24 hours in seconds


class RedisSessionStore:
    """Redis-backed chat session thread manager.

    Stores conversation history threads using Redis Lists under key pattern
    `sql_chatbot:session:{session_id}` with automatic TTL expiration.
    """

    def __init__(self, default_ttl: int = SESSION_TTL) -> None:
        """Initialize RedisSessionStore.

        Args:
            default_ttl (int, optional): Key time-to-live in seconds. Defaults to SESSION_TTL (86400).
        """
        self._ttl = default_ttl

    def _make_key(self, session_id: str) -> str:
        """Construct formatted Redis session key string.

        Args:
            session_id (str): Session identifier string.

        Returns:
            str: Formatted Redis key string.
        """
        return f"sql_chatbot:session:{session_id.strip()}"

    async def append_message_async(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Asynchronously append a message dict to the session history list in Redis.

        Args:
            session_id (str): Target session identifier string.
            role (str): Message role (`"user"` or `"assistant"`).
            content (str): Message text content string.
            metadata (Optional[Dict[str, Any]], optional): Optional metadata dict. Defaults to None.

        Returns:
            bool: True if message was successfully appended, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None or not session_id:
            return False

        try:
            key = self._make_key(session_id)
            msg_payload = {
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
            json_str = json.dumps(msg_payload)

            async with client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, json_str)
                pipe.expire(key, self._ttl)
                await pipe.execute()

            log.debug(f"RedisSessionStore: Appended '{role}' msg to session '{session_id}'")
            return True
        except Exception as e:
            log.warning(f"RedisSessionStore: append_message_async failed ({e})")
            return False

    async def get_history_async(
        self,
        session_id: str,
        limit: int = HISTORY_WINDOW,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve recent message history list for a session.

        Args:
            session_id (str): Session identifier string.
            limit (int, optional): Maximum number of recent messages to fetch. Defaults to HISTORY_WINDOW.

        Returns:
            List[Dict[str, Any]]: List of message dictionaries `[{"role": ..., "content": ...}]`.
        """
        client = redis_manager.get_client()
        if client is None or not session_id:
            return []

        try:
            key = self._make_key(session_id)
            raw_msgs = await client.lrange(key, -limit, -1)
            history = []
            for item in raw_msgs:
                try:
                    history.append(json.loads(item))
                except Exception:
                    pass
            return history
        except Exception as e:
            log.warning(f"RedisSessionStore: get_history_async failed ({e})")
            return []

    async def clear_session_async(self, session_id: str) -> bool:
        """Asynchronously delete session history key from Redis.

        Args:
            session_id (str): Session identifier string to clear.

        Returns:
            bool: True if session key was deleted, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None or not session_id:
            return False

        try:
            key = self._make_key(session_id)
            await client.delete(key)
            log.info(f"RedisSessionStore: Cleared session '{session_id}'")
            return True
        except Exception as e:
            log.warning(f"RedisSessionStore: clear_session_async failed ({e})")
            return False


# Singleton export
session_store = RedisSessionStore()
