"""
table_selector.py — Semantic table selection using ChromaDB vector similarity.
"""

# ── MODULE TAG: RAG Vector Table Selector (ChromaDB) ──
from typing import Any, Optional
from core.config.settings import (
    DEFAULT_EMBED_MODEL,
    VECTOR_RAG_THRESHOLD,
)
from core.config.logger import get_logger
from db.chromadb import get_table_schemas_collection
from core.llm.embedder import TextEmbedder
from db.indexer import index_tables

log = get_logger(__name__)

MOCK_THRESHOLD = 0.35


class TableSelector:
    """RAG-based selector that narrows down target tables based on ChromaDB vector query similarity."""

    def __init__(
        self,
        use_api_embeddings: bool = True,
        embed_model: str = DEFAULT_EMBED_MODEL,
        use_local_ollama: bool = True,
    ) -> None:
        self._embedder = TextEmbedder(embed_model=embed_model)

        if not (use_api_embeddings and use_local_ollama):
            self._embedder._api_available = False

        mode = (
            f"Gemini API ({embed_model})"
            if self._embedder._api_available
            else "Mock trigram mode"
        )
        log.debug(f"TableSelector: active embedding mode: {mode}")

    def select_tables(
        self,
        question:         str,
        candidate_tables: set,
        engine,
        threshold:        float | None = None,
    ) -> set[str]:
        """Narrows the table candidate list down to relevant tables using ChromaDB vector query."""
        if not candidate_tables:
            return set()

        if threshold is None:
            threshold = (
                VECTOR_RAG_THRESHOLD
                if self._embedder._api_available
                else MOCK_THRESHOLD
            )

        collection = get_table_schemas_collection()

        existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
        missing_tables = candidate_tables - existing_ids

        if missing_tables or collection.count() == 0:
            index_tables(candidate_tables, engine, self._embedder)

        q_vec = self._embedder.embed(question)
        candidate_list = list(candidate_tables)

        where_filter = (
            {"table_name": {"$in": candidate_list}}
            if len(candidate_list) > 1
            else {"table_name": {"$eq": candidate_list[0]}}
        )

        query_res = collection.query(
            query_embeddings=[q_vec],
            n_results=len(candidate_list),
            where=where_filter,
            include=["metadatas", "distances"]
        )

        matched: set[str] = set()

        if query_res and query_res.get("ids") and query_res["ids"][0]:
            retrieved_ids = query_res["ids"][0]
            distances = query_res["distances"][0] if query_res.get("distances") else []

            for tbl_id, dist in zip(retrieved_ids, distances):
                sim = round(1.0 - float(dist), 4)

                if sim >= threshold:
                    matched.add(tbl_id)

        if not matched:
            matched = set(candidate_tables)
            log.info("  VectorRAG: No tables met threshold — falling back to ALL candidate tables.")
        else:
            log.info(f"  VectorRAG: Selected {len(matched)} tables ({', '.join(sorted(matched))})")

        return matched

    def select_tables_with_score(
        self,
        question: str,
        candidate_tables: frozenset | set | list,
        engine: Any = None,
        threshold: Optional[float] = None,
    ) -> tuple[set[str], float]:
        """Select tables and return (matched_tables, top_vector_match_score)."""
        candidate_list = sorted(list(candidate_tables))
        if not candidate_list:
            return set(), 0.0

        if threshold is None:
            threshold = VECTOR_RAG_THRESHOLD

        q_vec = self._embedder.embed(question)
        collection = get_table_schemas_collection()

        where_filter = (
            {"table_name": {"$in": candidate_list}}
            if len(candidate_list) > 1
            else {"table_name": {"$eq": candidate_list[0]}}
        )

        query_res = collection.query(
            query_embeddings=[q_vec],
            n_results=len(candidate_list),
            where=where_filter,
            include=["metadatas", "distances"]
        )

        matched: set[str] = set()
        top_score: float = 0.0

        if query_res and query_res.get("ids") and query_res["ids"][0]:
            retrieved_ids = query_res["ids"][0]
            distances = query_res["distances"][0] if query_res.get("distances") else []

            for idx, (tbl_id, dist) in enumerate(zip(retrieved_ids, distances)):
                sim = round(1.0 - float(dist), 4)
                if idx == 0:
                    top_score = max(0.0, sim)

                if sim >= threshold:
                    matched.add(tbl_id)

        if not matched:
            matched = set(candidate_tables)
            if top_score == 0.0:
                top_score = 0.75

        return matched, top_score


    def clear_cache(self) -> None:
        """Clear all cached embeddings in ChromaDB 'table_schemas' collection."""
        try:
            collection = get_table_schemas_collection()
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            log.info("TableSelector: ChromaDB 'table_schemas' collection wiped.")
        except Exception as e:
            log.error(f"TableSelector: Failed to clear ChromaDB table_schemas collection: {e}")
