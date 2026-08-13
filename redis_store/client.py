"""
client.py — Centralized Redis connection manager for WMS AI Chatbot.

Provides async connection handling, support for REDIS_URL (Upstash/SSL) or standalone host/port,
and fallback handling if Redis is unreachable.
"""

# ── MODULE TAG: Centralized Redis Connection Manager ──
from typing import Optional
import redis.asyncio as aioredis

from core.config.settings import (
    REDIS_URL,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_PASSWORD,
    REDIS_DB,
    REDIS_ENABLED,
)
from core.config.logger import get_logger

log = get_logger(__name__)


class RedisClientManager:
    """Manages the connection to the Redis server for storing fast memory data.

    Attributes:
        _client (Optional[aioredis.Redis]): Active Redis connection client object or None.
        _connected (bool): True if currently connected to Redis.
        _enabled (bool): True if Redis is turned on in configuration.
    """

    def __init__(self) -> None:
        """Set up the Redis connection manager."""
        self._client: Optional[aioredis.Redis] = None
        self._connected: bool = False
        self._enabled: bool = REDIS_ENABLED

    async def connect(self) -> Optional[aioredis.Redis]:
        """Connect asynchronously to the Redis database server.

        Returns:
            Optional[aioredis.Redis]: Connected Redis client object, or None if connection fails.
        """
        if not self._enabled:
            log.info("RedisClientManager: REDIS_ENABLED is False — running in memory-only mode.")
            self._connected = False
            return None

        if self._connected and self._client is not None:
            return self._client

        try:
            if REDIS_URL:
                log.info("RedisClientManager: Connecting via REDIS_URL...")
                kwargs = {
                    "decode_responses": True,
                    "socket_connect_timeout": 3.0,
                    "socket_timeout": 5.0,
                    "max_connections": 50,
                }
                if REDIS_URL.startswith("rediss://"):
                    kwargs["ssl_cert_reqs"] = None

                self._client = aioredis.from_url(REDIS_URL, **kwargs)
            else:
                log.info(f"RedisClientManager: Connecting to {REDIS_HOST}:{REDIS_PORT} (DB {REDIS_DB})...")
                self._client = aioredis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    password=REDIS_PASSWORD or None,
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=1.0,
                    max_connections=50,
                )

            await self._client.ping()
            self._connected = True
            log.info("RedisClientManager: Connected successfully to Redis.")
            return self._client
        except Exception as e:
            log.warning(f"RedisClientManager: Failed to connect to Redis ({e}). Falling back to local memory mode.")
            self._client = None
            self._connected = False
            return None

    async def close(self) -> None:
        """Safely disconnect from the Redis server."""
        if self._client is not None:
            try:
                await self._client.close()
                log.info("RedisClientManager: Connection closed cleanly.")
            except Exception as e:
                log.warning(f"RedisClientManager: Error closing connection: {e}")
            finally:
                self._client = None
                self._connected = False

    def get_client(self) -> Optional[aioredis.Redis]:
        """Get the active Redis connection object.

        Returns:
            Optional[aioredis.Redis]: Active Redis client object, or None if offline.
        """
        return self._client if self._connected else None

    @property
    def is_connected(self) -> bool:
        """Check if the system is currently connected to Redis.

        Returns:
            bool: True if connected to Redis, False otherwise.
        """
        return self._connected and self._client is not None


# Export global singleton
redis_manager = RedisClientManager()
