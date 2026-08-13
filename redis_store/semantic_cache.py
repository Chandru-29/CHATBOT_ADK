"""
semantic_cache.py — Redis Semantic Vector Similarity Cache for WMS AI Chatbot.

Replaces SQLite ChromaDB to prevent file-locking bottlenecks under high concurrency (100+ users).
Supports RediSearch HNSW vector indexes (`FT.SEARCH`) and HASH vector cosine similarity calculations.
"""

# ── MODULE TAG: Redis Semantic Vector Cache ──
import json
import time
import uuid
import numpy as np
from typing import Optional, List, Dict, Any

from redis_store.client import redis_manager
from core.config.settings import SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_TTL
from core.config.logger import get_logger

log = get_logger(__name__)

INDEX_NAME = "idx:sql_chatbot_semantic"
PREFIX = "sql_chatbot:semantic:"


class RedisSemanticCache:
    """Redis-backed vector similarity cache supporting RediSearch & raw vector fallback.

    Attributes:
        _embedder: Text embedding generator instance.
        _threshold (float): Minimum cosine similarity score threshold for cache hits.
        _index_created (bool): Flag indicating if RediSearch vector index is active.
    """

    def __init__(
        self,
        embedder=None,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> None:
        """Initialize RedisSemanticCache instance.

        Args:
            embedder (optional): Text embedder instance. Defaults to None.
            threshold (float, optional): Cosine similarity threshold. Defaults to SEMANTIC_CACHE_THRESHOLD.
        """
        self._embedder = embedder
        self._threshold = threshold
        self._index_created = False

    async def init_index(self) -> bool:
        """Attempt to initialize RediSearch HNSW vector index in Redis.

        Returns:
            bool: True if RediSearch index exists or was created, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None:
            return False

        try:
            await client.execute_command("FT.INFO", INDEX_NAME)
            self._index_created = True
            return True
        except Exception as info_err:
            try:
                dim = getattr(self._embedder, "dim", 768) if self._embedder else 768
                await client.execute_command(
                    "FT.CREATE", INDEX_NAME,
                    "ON", "HASH",
                    "PREFIX", "1", PREFIX,
                    "SCHEMA",
                    "question", "TEXT",
                    "result_json", "TEXT",
                    "timestamp", "NUMERIC",
                    "vector", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(dim),
                    "DISTANCE_METRIC", "COSINE"
                )
                self._index_created = True
                log.info(f"RedisSemanticCache: Created RediSearch index '{INDEX_NAME}' (dim={dim}).")
                return True
            except Exception as create_err:
                log.debug(f"RedisSemanticCache: RediSearch index create bypassed ({create_err}). Falling back to hash vector scan.")
                self._index_created = False
                return False

    async def lookup_async(
        self,
        question: str,
        embedding: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Asynchronously perform vector similarity search in Redis.

        Args:
            question (str): Input user question string.
            embedding (Optional[List[float]], optional): Pre-computed vector embedding. Defaults to None.

        Returns:
            Optional[Dict[str, Any]]: Cached result dictionary if similarity >= threshold, else None.
        """
        client = redis_manager.get_client()
        if client is None:
            return None

        q_vec = list(embedding) if embedding is not None else (self._embedder.embed(question) if self._embedder else None)
        if not q_vec:
            return None

        q_arr = np.array(q_vec, dtype=np.float32)
        norm_q = np.linalg.norm(q_arr)
        if norm_q == 0:
            return None

        if self._index_created:
            try:
                blob = q_arr.tobytes()
                query_str = "*=>[KNN 1 @vector $vec AS score]"
                res = await client.execute_command(
                    "FT.SEARCH", INDEX_NAME,
                    query_str,
                    "PARAMS", "2", "vec", blob,
                    "SORTBY", "score", "ASC",
                    "RETURN", "3", "result_json", "timestamp", "score",
                    "DIALECT", "2"
                )
                if res and res[0] > 0 and len(res) >= 3:
                    fields = res[2]
                    field_dict = {}
                    for i in range(0, len(fields), 2):
                        k = fields[i]
                        v = fields[i + 1]
                        field_dict[k.decode("utf-8") if isinstance(k, bytes) else k] = (
                            v.decode("utf-8") if isinstance(v, bytes) else v
                        )

                    dist = float(field_dict.get("score", 1.0))
                    sim = round(1.0 - dist, 4)

                    if sim >= self._threshold:
                        ts = float(field_dict.get("timestamp", 0))
                        if SEMANTIC_CACHE_TTL > 0 and (time.time() - ts) > SEMANTIC_CACHE_TTL:
                            log.info("RedisSemanticCache: RediSearch entry expired.")
                            return None

                        result_json = field_dict.get("result_json", "")
                        if result_json:
                            log.info(f"REDIS SEMANTIC CACHE HIT (RediSearch sim={sim:.4f})")
                            return json.loads(result_json)
            except Exception as e:
                log.debug(f"RedisSemanticCache: FT.SEARCH failed ({e}) — trying scan fallback.")

        try:
            keys = await client.keys(f"{PREFIX}*")
            if not keys:
                return None

            best_sim = -1.0
            best_result = None

            for key in keys[:50]:
                hdata = await client.hgetall(key)
                if not hdata:
                    continue

                ts = float(hdata.get("timestamp", 0))
                if SEMANTIC_CACHE_TTL > 0 and (time.time() - ts) > SEMANTIC_CACHE_TTL:
                    await client.delete(key)
                    continue

                vec_bytes = hdata.get("vector_raw")
                if vec_bytes:
                    stored_vec = np.frombuffer(vec_bytes if isinstance(vec_bytes, bytes) else vec_bytes.encode('latin1'), dtype=np.float32)
                    norm_s = np.linalg.norm(stored_vec)
                    if norm_s > 0:
                        sim = float(np.dot(q_arr, stored_vec) / (norm_q * norm_s))
                        if sim > best_sim:
                            best_sim = sim
                            best_result = hdata.get("result_json")

            if best_sim >= self._threshold and best_result:
                log.info(f"REDIS SEMANTIC CACHE HIT (Vector scan sim={best_sim:.4f})")
                return json.loads(best_result)

        except Exception as scan_err:
            log.warning(f"RedisSemanticCache: lookup scan failed: {scan_err}")

        return None

    async def store_async(
        self,
        question: str,
        result: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Asynchronously store query vector and result JSON in Redis.

        Args:
            question (str): User question string.
            result (Dict[str, Any]): Result payload dictionary to store.
            embedding (Optional[List[float]], optional): Pre-computed vector embedding. Defaults to None.

        Returns:
            bool: True if entry stored successfully, False otherwise.
        """
        client = redis_manager.get_client()
        if client is None:
            return False

        q_vec = list(embedding) if embedding is not None else (self._embedder.embed(question) if self._embedder else None)
        if not q_vec:
            return False

        try:
            doc_id = f"{PREFIX}{uuid.uuid4().hex}"
            q_arr = np.array(q_vec, dtype=np.float32)
            blob = q_arr.tobytes()
            result_json = json.dumps(result)

            await client.hset(
                doc_id,
                mapping={
                    "question": question[:500],
                    "result_json": result_json,
                    "timestamp": str(time.time()),
                    "vector": blob,
                    "vector_raw": blob,
                }
            )
            if SEMANTIC_CACHE_TTL > 0:
                await client.expire(doc_id, SEMANTIC_CACHE_TTL)

            log.debug(f"RedisSemanticCache: Stored vector entry '{doc_id}'")
            return True
        except Exception as e:
            log.warning(f"RedisSemanticCache: store_async failed: {e}")
            return False


# Global export instance
redis_semantic_cache = RedisSemanticCache()
