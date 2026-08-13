"""
semantic_cache.py — ChromaDB-backed semantic similarity cache for the chatbot pipeline.
"""

# ── MODULE TAG: Semantic Response Cache (ChromaDB) ──
import json
import time
import uuid
from typing import Optional

from core.config.settings import SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_TTL
from core.config.logger import get_logger
from db.chromadb import get_semantic_cache_collection, reset_semantic_cache_collection

log = get_logger(__name__)


def clear_semantic_cache_chroma() -> None:
    """Wipe and recreate the ChromaDB vector semantic cache memory."""
    try:
        reset_semantic_cache_collection()
        log.info("SemanticCache: ChromaDB 'semantic_cache' collection wiped and reset.")
    except Exception as e:
        log.error(f"SemanticCache: Failed to clear ChromaDB semantic_cache collection: {e}")


class SemanticCache:
    """Saves past answers in a local vector database to answer similar questions instantly.

    Attributes:
        _embedder: Tool used to convert questions into numbers for comparison.
        _threshold (float): Minimum similarity score needed to consider an answer a match.
        _enabled (bool): True if vector semantic caching is active, False otherwise.
    """

    def __init__(
        self,
        embedder,
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> None:
        """Set up the SemanticCache tool.

        Args:
            embedder: Tool used to convert text to vectors.
            threshold (float, optional): Similarity match score cutoff. Defaults to SEMANTIC_CACHE_THRESHOLD.
        """
        self._embedder  = embedder
        self._threshold = threshold
        self._enabled   = getattr(embedder, "_api_available", getattr(embedder, "_ollama_available", False))

        if not self._enabled:
            log.warning(
                "SemanticCache: API embeddings unavailable — "
                "semantic cache DISABLED."
            )
        else:
            log.info(f"SemanticCache: enabled via ChromaDB — threshold={threshold}")

    def _get_collection(self):
        """Get the active ChromaDB memory folder for storing answers.

        Returns:
            Collection: Active vector collection object.
        """
        try:
            return get_semantic_cache_collection()
        except Exception:
            return reset_semantic_cache_collection()

    def _reset_collection(self, reason: str):
        """Wipe and recreate the vector database collection if an error happens.

        Args:
            reason (str): Explanation for resetting memory.

        Returns:
            Collection: Clean vector collection object or None.
        """
        log.warning(f"SemanticCache: auto-resetting collection — {reason}")
        try:
            col = reset_semantic_cache_collection()
            log.warning("SemanticCache: ChromaDB collection wiped and recreated.")
            return col
        except Exception as wipe_err:
            log.error(f"SemanticCache: failed to wipe collection during reset: {wipe_err}")
            self._enabled = False
            return None

    def lookup(
        self,
        question: str,
        embedding: Optional[list[float]] = None,
    ) -> Optional[dict]:
        """Search the vector database for a past question with a similar meaning.

        Args:
            question (str): The user's question.
            embedding (Optional[list[float]], optional): Pre-calculated vector for the question. Defaults to None.

        Returns:
            Optional[dict]: The saved answer dictionary if a similar match is found, otherwise None.
        """
        if not self._enabled:
            return None

        col = self._get_collection()

        try:
            if col.count() == 0:
                return None
        except Exception as e:
            col = self._reset_collection(f"Collection count() access failed ({e})")
            return None

        try:
            q_vec = list(embedding) if embedding is not None else self._embedder.embed(question)
        except Exception as e:
            log.warning(f"SemanticCache: embed failed on lookup ({e}) — cache miss.")
            return None

        try:
            query_res = col.query(
                query_embeddings=[q_vec],
                n_results=1,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            self._reset_collection(f"query() failed: {e}")
            return None

        if query_res and query_res.get("ids") and query_res["ids"][0]:
            dist = query_res["distances"][0][0]
            sim = round(1.0 - float(dist), 4)

            if sim >= self._threshold:
                metadatas = query_res.get("metadatas")
                metadata = metadatas[0][0] if metadatas and metadatas[0] and metadatas[0][0] else None
                if not metadata or not isinstance(metadata, dict):
                    return None

                entry_time = metadata.get("timestamp", 0)
                if SEMANTIC_CACHE_TTL > 0 and (time.time() - entry_time) > SEMANTIC_CACHE_TTL:
                    log.info(f"SemanticCache: entry expired (age={time.time() - entry_time:.1f}s > {SEMANTIC_CACHE_TTL}s) for: '{question[:80]}'")
                    try:
                        col.delete(ids=[query_res["ids"][0][0]])
                    except Exception:
                        pass
                    return None

                result_json = metadata.get("result_json", "")
                if result_json:
                    try:
                        cached_result = json.loads(result_json)
                        log.info(f"SEMANTIC CACHE HIT (ChromaDB score={sim:.4f}) for: '{question[:80]}'")
                        return cached_result
                    except Exception as e:
                        log.error(f"SemanticCache: Failed to deserialize cached result JSON: {e}")

        return None

    def store(
        self,
        question: str,
        result: dict,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Save a new question and its answer into the vector database.

        Args:
            question (str): The question text.
            result (dict): The answer dictionary payload.
            embedding (Optional[list[float]], optional): Pre-calculated vector for the question. Defaults to None.
        """
        if not self._enabled:
            return

        try:
            q_vec = list(embedding) if embedding is not None else self._embedder.embed(question)
        except Exception as e:
            log.warning(f"SemanticCache: embed failed on store ({e}) — skipping.")
            return

        doc_id = str(uuid.uuid4())
        col = self._get_collection()
        try:
            result_json = json.dumps(result)
            col.upsert(
                ids=[doc_id],
                embeddings=[q_vec],
                documents=[question],
                metadatas=[{
                    "question": question[:500],
                    "result_json": result_json,
                    "timestamp": time.time(),
                }]
            )
            log.debug(f"SemanticCache (ChromaDB): stored entry for '{question[:80]}' — total count={col.count()}")
        except Exception as e:
            log.warning(f"SemanticCache: Collection operation failed on store ({e}). Recreating collection...")
            try:
                col = self._reset_collection(f"Store retry after exception: {e}")
                if col is not None:
                    col.upsert(
                        ids=[doc_id],
                        embeddings=[q_vec],
                        documents=[question],
                        metadatas=[{
                            "question": question[:500],
                            "result_json": result_json,
                            "timestamp": time.time(),
                        }]
                    )
                    log.info(f"SemanticCache (ChromaDB): Successfully recreated collection and stored entry with vector dimension {len(q_vec)}.")
                    return
            except Exception as retry_err:
                log.error(f"SemanticCache: Failed to store entry after recreating collection: {retry_err}")

    def remove(self, result: dict) -> None:
        """Remove a cached answer from the vector database.

        Args:
            result (dict): Answer dictionary to remove.
        """
        if not self._enabled:
            return
        log.info("SemanticCache: eviction requested.")

    def clear(self) -> None:
        """Clear all saved answers from the vector database."""
        clear_semantic_cache_chroma()


