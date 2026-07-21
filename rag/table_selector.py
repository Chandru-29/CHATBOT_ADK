"""
table_selector.py — Semantic table selection using dense vector similarity.
"""


# ── MODULE TAG: RAG Vector Table Selector ──
from cachetools import TTLCache
from cachetools.keys import hashkey

from config.settings import (
    DEFAULT_EMBED_MODEL,
    OLLAMA_THRESHOLD,
    MOCK_THRESHOLD,
    REGISTRY_TTL,
)
from config.logger import get_logger
from rag.embedder import TextEmbedder
from rag.indexer import index_tables
from rag.similarity import cosine_sim

log = get_logger(__name__)


class TableSelector:
    """
    RAG-based selector that narrows down target tables based on user query similarity.
    """

    def __init__(
        self,
        use_local_ollama: bool = True,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self._embedder        = TextEmbedder(embed_model=embed_model)
        self._registry_cache  = TTLCache(maxsize=5, ttl=REGISTRY_TTL)

        if not use_local_ollama:
            self._embedder._ollama_available = False

        mode = (
            f"Ollama ({embed_model})"
            if self._embedder._ollama_available
            else "Mock trigram mode"
        )
        log.info(f"TableSelector: active embedding mode: {mode}")

    def select_tables(
        self,
        question:         str,
        candidate_tables: set,
        engine,
        threshold:        float | None = None,
    ) -> set[str]:
        """
        Narrows the table candidate list down to only relevant ones.
        """
        # ── SELECT SIMILARITY CUTOFF THRESHOLD ──────────────────────────────────────────
        if threshold is None:
            threshold = (
                OLLAMA_THRESHOLD
                if self._embedder._ollama_available
                else MOCK_THRESHOLD
            )

        # ── RETRIEVE OR GENERATE VECTOR REGISTRY ─────────────────────────────────────────
        frozen    = frozenset(candidate_tables)
        cache_key = hashkey(frozen)

        if cache_key not in self._registry_cache:
            self._registry_cache[cache_key] = index_tables(
                candidate_tables, engine, self._embedder
            )

        registry = self._registry_cache[cache_key]

        # ── CALCULATE COSINE SIMILARITY & FILTER TABLES ──────────────────────────────────
        q_vec = self._embedder.embed(question)
        similarities: dict[str, float] = {}
        matched:      set[str]         = set()

        for table, t_vec in registry.items():
            sim = round(cosine_sim(q_vec, t_vec), 4)
            similarities[table] = sim
            if sim >= threshold:
                matched.add(table)

        log.info(f"TableSelector: margins={similarities} -> selected={matched}")


        # Fallback if zero matches are found
        if not matched:
            log.info(f"TableSelector: no tables above {threshold} - using entire pool")
            return candidate_tables

        return matched


# Backward-compatibility aliases
VectorSchemaRAG = TableSelector
TableSelector.retrieve_schemas = TableSelector.select_tables
