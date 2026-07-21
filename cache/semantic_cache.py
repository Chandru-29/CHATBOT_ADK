"""
semantic_cache.py — Semantic similarity cache for the chatbot pipeline.

Instead of exact-string matching, incoming questions are embedded with
nomic-embed-text and compared to stored question embeddings via cosine
similarity. If any stored question scores ≥ threshold, the cached result
is returned immediately — skipping rephrase, intent, RAG, and agent_loop.

Falls back to the existing exact-text TTLCache when Ollama embeddings are
unavailable (mock trigram cosine similarity is too coarse for high thresholds).

Usage:
    sem_cache = SemanticCache(embedder)
    hit = sem_cache.lookup(question)       # returns result dict or None
    sem_cache.store(question, result)      # stores after a full pipeline run
"""


# ── MODULE TAG: Semantic Response Cache ──
# ── STITCHGUARD LAYER: L5 (Cached Data Redactions) ──
import math
import time
from typing import Optional

from config.settings import SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_TTL, SEMANTIC_CACHE_MAX
from config.logger import get_logger

log = get_logger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Fast inline cosine similarity — avoids importing rag.similarity for a clean dep graph."""
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a)) or 1.0
    nb   = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class SemanticCache:
    """
    TTL-bounded semantic similarity cache.

    Stores (question_embedding, result, expiry_timestamp) tuples in a list.
    On lookup, embeds the incoming question and compares cosine similarity
    against all stored embeddings. First hit at or above the threshold wins.

    Eviction: expired entries are pruned lazily on every lookup/store call.
    Max size: oldest entries are dropped when the cache exceeds SEMANTIC_CACHE_MAX.

    Parameters
    ----------
    embedder : TextEmbedder
        The shared TextEmbedder instance (Ollama or mock trigram).
    threshold : float
        Cosine similarity cutoff (default from settings: 0.92).
    ttl : int
        Seconds before an entry expires (default from settings: 300).
    max_size : int
        Maximum number of entries before oldest are evicted (default: 200).
    """

    def __init__(
        self,
        embedder,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
        ttl:       int   = SEMANTIC_CACHE_TTL,
        max_size:  int   = SEMANTIC_CACHE_MAX,
    ) -> None:
        self._embedder  = embedder
        self._threshold = threshold
        self._ttl       = ttl
        self._max_size  = max_size
        # Each entry: {"vec": [...], "result": {...}, "expires": float}
        self._entries: list[dict] = []
        self._enabled = embedder._ollama_available
        if not self._enabled:
            log.warning(
                "SemanticCache: Ollama embeddings unavailable — "
                "semantic cache DISABLED, falling back to exact-text TTLCache only."
            )
        else:
            log.info(
                f"SemanticCache: enabled — threshold={threshold}, "
                f"ttl={ttl}s, max={max_size} entries"
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        now = time.monotonic()
        self._entries = [e for e in self._entries if e["expires"] > now]

    def _evict_overflow(self) -> None:
        if len(self._entries) > self._max_size:
            # Drop oldest (front of list)
            self._entries = self._entries[-self._max_size:]

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(self, question: str) -> Optional[dict]:
        """
        Embed *question* and scan stored embeddings for a cosine-similarity hit.

        Returns:
            The cached result dict if a match is found, otherwise None.
        """
        if not self._enabled:
            return None

        # ── MONOTONIC CLOCK EXPIRY CHECK & PRUNING ──────────────────────────────
        self._evict_expired()
        if not self._entries:
            return None

        # ── GENERATE QUERY VECTOR EMBEDDING ─────────────────────────────────────
        try:
            q_vec = self._embedder.embed(question)
        except Exception as e:
            log.warning(f"SemanticCache: embed failed on lookup ({e}) — cache miss.")
            return None

        # ── SCAN REGISTERED VECTOR EMBEDDINGS FOR SIMILARITY MATCH ──────────────
        best_score = 0.0
        best_result = None
        for entry in self._entries:
            score = _cosine(q_vec, entry["vec"])
            if score >= self._threshold and score > best_score:
                best_score  = score
                best_result = entry["result"]

        if best_result is not None:
            log.info(f"SEMANTIC CACHE HIT (score={best_score:.4f}) for: '{question[:80]}'")
            return best_result

        return None

    def store(self, question: str, result: dict) -> None:
        """
        Embed *question* and store it alongside *result*.
        Silently skips if embeddings are unavailable or embed fails.
        """
        if not self._enabled:
            return

        # ── GENERATE VECTOR EMBEDDING FOR STORING ────────────────────────────────
        try:
            q_vec = self._embedder.embed(question)
        except Exception as e:
            log.warning(f"SemanticCache: embed failed on store ({e}) — skipping.")
            return

        # ── PRUNE & ENQUEUE FRESH ENTRY WITH TTL STAMP ───────────────────────────
        self._evict_expired()
        self._entries.append({
            "vec":     q_vec,
            "result":  result,
            "expires": time.monotonic() + self._ttl,
        })
        self._evict_overflow()
        log.debug(f"SemanticCache: stored entry for '{question[:80]}' — size={len(self._entries)}")


    def remove(self, result: dict) -> None:
        """Remove any entry with the matching result dictionary reference."""
        if not self._enabled:
            return
        initial_len = len(self._entries)
        self._entries = [e for e in self._entries if e["result"] is not result]
        log.info(f"SemanticCache: evicted {initial_len - len(self._entries)} entry/entries from cache.")

