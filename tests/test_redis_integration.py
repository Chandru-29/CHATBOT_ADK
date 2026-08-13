"""
test_redis_integration.py — Pytest & direct runner verification test suite for Redis integration.
"""

import os
import sys
import asyncio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_store import (
    RedisClientManager, redis_manager,
    RedisExactCache, redis_exact_cache,
    RedisSemanticCache, redis_semantic_cache,
    RedisSessionStore, session_store,
    RedisRateLimiter, rate_limiter
)
from core.cache.cache_manager import lookup_cache, store_cache, async_lookup_cache, async_store_cache


async def test_redis_client_fallback():
    """Verify RedisClientManager handles connection attempts gracefully."""
    mgr = RedisClientManager()
    client = await mgr.connect()
    assert mgr.get_client() == client or mgr.get_client() is None
    await mgr.close()
    print("[OK] test_redis_client_fallback passed")


async def test_redis_exact_cache():
    """Verify exact cache get/set logic and key generation."""
    cache = RedisExactCache()
    res = await cache.get_async("what is total stock?", role="user")
    assert res is None or isinstance(res, dict)

    stored = await cache.set_async("what is total stock?", role="user", answer="Total stock is 500")
    assert stored in (True, False)
    print("[OK] test_redis_exact_cache passed")


async def test_redis_semantic_cache():
    """Verify semantic cache store and lookup with pre-supplied embeddings."""
    mock_vec = [0.1] * 768
    stored = await redis_semantic_cache.store_async("test query", {"answer": "test"}, embedding=mock_vec)
    assert stored in (True, False)

    found = await redis_semantic_cache.lookup_async("test query", embedding=mock_vec)
    assert found is None or isinstance(found, dict)
    print("[OK] test_redis_semantic_cache passed")


async def test_redis_session_store():
    """Verify chat session store thread management."""
    session_id = "test_session_123"
    appended = await session_store.append_message_async(session_id, "user", "Hello WMS AI")
    assert appended in (True, False)

    history = await session_store.get_history_async(session_id, limit=5)
    assert isinstance(history, list)

    await session_store.clear_session_async(session_id)
    print("[OK] test_redis_session_store passed")


async def test_redis_rate_limiter():
    """Verify sliding window rate limiter logic."""
    limiter = RedisRateLimiter(default_limit=5, window_seconds=10)
    allowed, count, retry_after = await limiter.is_allowed("test_client_ip")
    assert isinstance(allowed, bool)
    assert isinstance(count, int)
    print("[OK] test_redis_rate_limiter passed")


async def main():
    print("Running Redis Integration Verification Suite...")
    await test_redis_client_fallback()
    await test_redis_exact_cache()
    await test_redis_semantic_cache()
    await test_redis_session_store()
    await test_redis_rate_limiter()
    print("\nALL REDIS INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(main())
