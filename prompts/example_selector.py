"""
example_selector.py — Persistent ChromaDB Vector RAG Selector for Few-Shot Exemplars.

Stores pre-computed exemplar question embeddings in ChromaDB collection 'fewshot_exemplars'
with SHA-256 metadata hash invalidation for zero-latency runtime query matching.
"""

# ── MODULE TAG: RAG Few-Shot Example Selector (ChromaDB) ──
import json
import hashlib
from typing import Optional, List, Dict, Any, Set

from core.config.settings import DEFAULT_EMBED_MODEL
from core.config.logger import get_logger
from core.llm.embedder import TextEmbedder
from db.chromadb import (
    get_fewshot_exemplars_collection,
    reset_fewshot_exemplars_collection,
)

log = get_logger(__name__)

DEFAULT_TOP_K = 5


class ExampleSelector:
    """ChromaDB vector store backed selector computing embeddings for Q/A exemplars in examples.yml.

    Attributes:
        _embedder: TextEmbedder model instance.
    """

    def __init__(self, embedder: Optional[TextEmbedder] = None) -> None:
        """Initialize ExampleSelector instance.

        Args:
            embedder (Optional[TextEmbedder], optional): TextEmbedder instance. Defaults to None.
        """
        self._embedder = embedder or TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)

    @staticmethod
    def _compute_examples_hash(raw_examples_dict: Dict[str, Any]) -> str:
        """Compute SHA-256 hash digest of raw examples dictionary to detect content changes.

        Args:
            raw_examples_dict (Dict[str, Any]): Dictionary of intent -> list of Q/A exemplars.

        Returns:
            str: SHA-256 hexadecimal string digest.
        """
        serialized = json.dumps(raw_examples_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def precompute_embeddings(self, raw_examples_dict: Dict[str, Any], force_recompute: bool = False) -> None:
        """Pre-compute and index all exemplar question vector embeddings into ChromaDB.

        Args:
            raw_examples_dict (Dict[str, Any]): Dictionary mapping domain intents to lists of Q/A exemplars.
            force_recompute (bool, optional): If True, forces collection re-indexing. Defaults to False.
        """
        if not raw_examples_dict:
            return

        curr_hash = self._compute_examples_hash(raw_examples_dict)
        collection = get_fewshot_exemplars_collection()

        stored_meta = collection.metadata or {}
        stored_hash = stored_meta.get("hash")

        if not force_recompute and stored_hash == curr_hash and collection.count() > 0:
            log.info(
                f"ExampleSelector: ChromaDB collection 'fewshot_exemplars' hash matched "
                f"({curr_hash[:8]}...). Count={collection.count()}. Skipping re-indexing."
            )
            return

        log.info("ExampleSelector: Re-indexing exemplars into ChromaDB collection 'fewshot_exemplars'...")
        collection = reset_fewshot_exemplars_collection(metadata={"hash": curr_hash})

        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for intent, raw_examples in raw_examples_dict.items():
            if not isinstance(raw_examples, list):
                continue

            for idx, ex in enumerate(raw_examples):
                if not isinstance(ex, dict):
                    continue
                q = ex.get("q", "").strip()
                a = ex.get("a", "").strip()
                if not q or not a:
                    continue

                vec = self._embedder.embed(q)
                doc_id = hashlib.md5(f"{intent}_{idx}_{q}".encode("utf-8")).hexdigest()

                documents.append(q)
                embeddings.append(vec)
                metadatas.append({"intent": intent, "q": q, "a": a})
                ids.append(doc_id)

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            log.info(f"ExampleSelector: Upserted {len(ids)} exemplars into ChromaDB 'fewshot_exemplars'.")

    def select_top_k_examples(
        self,
        question: str,
        intent: str,
        raw_examples: List[Dict[str, str]],
        top_k: Optional[int] = None,
        focused_tables: Optional[Set[str]] = None,
    ) -> str:
        """Rank exemplars using ChromaDB vector cosine similarity and return formatted Markdown string.

        Args:
            question (str): User question string.
            intent (str): Domain intent string.
            raw_examples (List[Dict[str, str]]): Raw Q/A exemplars list.
            top_k (Optional[int], optional): Top-K limit override. Defaults to None.
            focused_tables (Optional[Set[str]], optional): Grounded tables set. Defaults to None.

        Returns:
            str: Formatted Markdown query examples block string.
        """
        if not raw_examples:
            return ""

        effective_k = top_k if top_k is not None else DEFAULT_TOP_K
        collection = get_fewshot_exemplars_collection()

        # If ChromaDB collection is empty, run auto pre-computation
        if collection.count() == 0 and raw_examples:
            self.precompute_embeddings({intent: raw_examples})
            collection = get_fewshot_exemplars_collection()

        q_vec = self._embedder.embed(question)

        try:
            results = collection.query(
                query_embeddings=[q_vec],
                n_results=effective_k,
                where={"intent": intent},
            )
            matched_meta = results.get("metadatas", [[]])[0] if results else []
        except Exception as e:
            log.warning(f"ExampleSelector: ChromaDB query failed ({e}). Falling back to raw list.")
            matched_meta = raw_examples[:effective_k]

        if not matched_meta:
            return ""

        log.info(
            f"ExampleSelector: Selected top-{len(matched_meta)} (adaptive_k={effective_k}) exemplars "
            f"for '{question[:50]}'"
        )

        lines = [f"Relevant Query Examples (Top-{len(matched_meta)}):"]
        for ex in matched_meta:
            q = ex.get("q", "").strip() if isinstance(ex, dict) else ""
            a = ex.get("a", "").strip() if isinstance(ex, dict) else ""
            if q and a:
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")
                lines.append("")

        return "\n".join(lines).strip()

    def clear_cache(self) -> None:
        """Wipe and reset ChromaDB 'fewshot_exemplars' collection."""
        reset_fewshot_exemplars_collection()
        log.info("ExampleSelector: ChromaDB 'fewshot_exemplars' collection wiped.")


# Singleton instance
example_selector = ExampleSelector()
