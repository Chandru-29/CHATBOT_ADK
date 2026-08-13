"""
rate_limiter.py — Redis Sliding Window Rate Limiter & FastAPI Middleware.

Implements a Redis-backed sliding-window rate limiter using Redis sorted sets (ZADD, ZREMRANGEBYSCORE)
to prevent API 429 rate limit issues under high concurrency (100+ users).
"""

# ── MODULE TAG: Distributed Redis Rate Limiter ──
import time
from typing import Tuple, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from redis_store.client import redis_manager
from core.config.settings import REDIS_RATE_LIMIT_PER_MIN
from core.config.logger import get_logger

log = get_logger(__name__)


class RedisRateLimiter:
    """Sliding-window rate limiter using Redis sorted sets.

    Attributes:
        default_limit (int): Maximum request limit within the sliding window.
        window_seconds (int): Sliding window duration in seconds.
    """

    def __init__(self, default_limit: int = REDIS_RATE_LIMIT_PER_MIN, window_seconds: int = 60) -> None:
        """Initialize RedisRateLimiter instance.

        Args:
            default_limit (int, optional): Maximum allowed requests. Defaults to REDIS_RATE_LIMIT_PER_MIN.
            window_seconds (int, optional): Window size in seconds. Defaults to 60.
        """
        self.default_limit = default_limit
        self.window_seconds = window_seconds

    async def is_allowed(
        self,
        identifier: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> Tuple[bool, int, float]:
        """Check if request from identifier is allowed within sliding window.

        Args:
            identifier (str): Client IP address or request identifier string.
            max_requests (Optional[int], optional): Request limit override. Defaults to None.
            window_seconds (Optional[int], optional): Window duration override. Defaults to None.

        Returns:
            Tuple[bool, int, float]: A tuple containing:
                - bool: True if request is allowed, False if limit exceeded.
                - int: Current request count within window.
                - float: Retry-after duration in seconds if limited.
        """
        client = redis_manager.get_client()
        if client is None:
            return True, 0, 0.0

        limit = max_requests or self.default_limit
        window = window_seconds or self.window_seconds
        now = time.time()
        window_start = now - window
        key = f"sql_chatbot:ratelimit:{identifier.strip()}"
        member_id = f"{now}"

        try:
            async with client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {member_id: now})
                pipe.zcard(key)
                pipe.expire(key, window + 5)
                results = await pipe.execute()

            request_count = results[2]
            if request_count > limit:
                log.warning(f"RedisRateLimiter: Rate limit exceeded for '{identifier}' ({request_count}/{limit})")
                return False, request_count, float(window)

            return True, request_count, 0.0
        except Exception as e:
            log.warning(f"RedisRateLimiter: Check failed ({e}) — allowing request fallback.")
            return True, 0, 0.0


# Singleton instance
rate_limiter = RedisRateLimiter()


class RedisRateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware enforcing client IP rate limiting."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process HTTP request through rate limiting filter.

        Args:
            request (Request): Incoming FastAPI HTTP Request object.
            call_next (Callable): Next middleware/route handler coroutine.

        Returns:
            Response: FastAPI Response object or 429 Too Many Requests JSONResponse.
        """
        if request.url.path in ("/status", "/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
        client_ip = client_ip.split(",")[0].strip()

        allowed, current_count, retry_after = await rate_limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too Many Requests. Please slow down and try again later.",
                    "error": "Rate limit exceeded",
                    "client_ip": client_ip,
                    "retry_after_seconds": int(retry_after),
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(rate_limiter.default_limit),
                    "X-RateLimit-Remaining": "0",
                }
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.default_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limiter.default_limit - current_count))
        return response
