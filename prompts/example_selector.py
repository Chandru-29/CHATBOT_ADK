"""
example_selector.py — Vector Similarity RAG Selector for Few-Shot Query Exemplars.

Dynamically selects the top-K (default 5) most semantically relevant Q/A
exemplars matching the user's question, reducing system prompt token size by 80%.
"""

# ── MODULE TAG: RAG Few-Shot Example Selector ──
import math
from typing import Optional, List, Dict, Any

from core.config.settings import DEFAULT_EMBED_MODEL
from core.config.logger import get_logger
from core.llm.embedder import TextEmbedder

log = get_logger(__name__)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1)) or 1.0
    mag2 = math.sqrt(sum(b * b for b in vec2)) or 1.0
    return dot / (mag1 * mag2)


class ExampleSelector:
    """
    RAG-backed selector that computes vector embeddings for all Q/A exemplars in examples.yml
    and retrieves top-K most semantically relevant examples for a user question.
    """

    def __init__(self, embedder: Optional[TextEmbedder] = None) -> None:
        self._embedder = embedder or TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)
        self._exemplar_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _index_examples(self, intent: str, raw_examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Compute and cache embedding vectors for all exemplar questions of an intent."""
        indexed = []
        for ex in raw_examples:
            q = ex.get("q", "").strip()
            a = ex.get("a", "").strip()
            if not q or not a:
                continue
            vec = self._embedder.embed(q)
            indexed.append({
                "q": q,
                "a": a,
                "vector": vec,
            })
        log.info(f"ExampleSelector: Indexed {len(indexed)} exemplars for intent '{intent}'")
        return indexed

    def select_top_k_examples(
        self,
        question: str,
        intent: str,
        raw_examples: List[Dict[str, str]],
        top_k: int = 5,
    ) -> str:
        """
        Rank exemplars by vector cosine similarity to *question* and return formatted
        top-K Q/A pairs.
        """
        if not raw_examples:
            return ""

        if intent not in self._exemplar_cache or len(self._exemplar_cache[intent]) != len(raw_examples):
            self._exemplar_cache[intent] = self._index_examples(intent, raw_examples)

        exemplars = self._exemplar_cache[intent]
        if not exemplars:
            return ""

        q_vec = self._embedder.embed(question)

        scored = []
        for ex in exemplars:
            score = _cosine_similarity(q_vec, ex["vector"])
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored[:top_k]

        log.info(
            f"ExampleSelector: Selected top-{len(top_matches)} exemplars for '{question[:50]}' "
            f"(sim range: {top_matches[0][0]:.3f} to {top_matches[-1][0]:.3f})"
        )

        lines = [f"Relevant Query Examples (Top-{len(top_matches)}):"]
        for _, ex in top_matches:
            lines.append(f"Q: {ex['q']}")
            lines.append(f"A: {ex['a']}")
            lines.append("")

        return "\n".join(lines).strip()

    def clear_cache(self) -> None:
        """Clear cached exemplar vectors."""
        self._exemplar_cache.clear()
        log.info("ExampleSelector: Exemplar vector cache cleared.")


# Singleton instance
example_selector = ExampleSelector()
