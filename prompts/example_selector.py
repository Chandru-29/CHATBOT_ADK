"""
example_selector.py — Vector Similarity RAG Selector for Few-Shot Query Exemplars.

Dynamically selects the top-K (default 5) most semantically relevant Q/A
exemplars matching the user's question, reducing system prompt token size by 80%.
"""

# ── MODULE TAG: RAG Few-Shot Example Selector ──
import math
from typing import Optional, List, Dict, Any, Set

from core.config.settings import DEFAULT_EMBED_MODEL
from core.config.logger import get_logger
from core.llm.embedder import TextEmbedder

log = get_logger(__name__)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity score between two float vectors.

    Args:
        vec1 (list[float]): First vector.
        vec2 (list[float]): Second vector.

    Returns:
        float: Cosine similarity score value.
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1)) or 1.0
    mag2 = math.sqrt(sum(b * b for b in vec2)) or 1.0
    return dot / (mag1 * mag2)


class ExampleSelector:
    """RAG-backed selector computing vector embeddings for Q/A exemplars in examples.yml.

    Attributes:
        _embedder: TextEmbedder model instance.
        _exemplar_cache: Cache mapping intent strings to indexed exemplar dicts.
    """

    def __init__(self, embedder: Optional[TextEmbedder] = None) -> None:
        """Initialize ExampleSelector instance.

        Args:
            embedder (Optional[TextEmbedder], optional): TextEmbedder instance. Defaults to None.
        """
        self._embedder = embedder or TextEmbedder(embed_model=DEFAULT_EMBED_MODEL)
        self._exemplar_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _index_examples(self, intent: str, raw_examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Compute and cache embedding vectors for all exemplar questions of an intent.

        Args:
            intent (str): Intent domain string.
            raw_examples (List[Dict[str, str]]): List of raw Q/A exemplar dicts.

        Returns:
            List[Dict[str, Any]]: List of indexed exemplar dicts with pre-computed vector embeddings.
        """
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

    def determine_adaptive_k(self, question: str, focused_tables: Optional[Set[str]] = None) -> int:
        """Dynamically determine optimal Top-K exemplar limit based on query syntax complexity.

        Args:
            question (str): User question string.
            focused_tables (Optional[Set[str]], optional): Grounded table scope set. Defaults to None.

        Returns:
            int: Calculated K limit (1 for simple, 3 for intermediate, 5 for complex).
        """
        q_lower = question.lower().strip()
        tables_count = len(focused_tables) if focused_tables else 1

        complex_indicators = ["most", "least", "highest", "lowest", "compare", "chain", "activity log", "between"]
        if any(ci in q_lower for ci in complex_indicators) or tables_count >= 3:
            return 5

        intermediate_indicators = ["join", "group by", "average", "avg", "per warehouse", "status breakdown", "mapping", "vendor"]
        if any(ii in q_lower for ii in intermediate_indicators) or tables_count == 2:
            return 3

        simple_starts = ["how many", "count", "list", "show all", "what is the status", "total"]
        if any(q_lower.startswith(s) for s in simple_starts) or tables_count <= 1:
            return 1

        return 3

    def select_top_k_examples(
        self,
        question: str,
        intent: str,
        raw_examples: List[Dict[str, str]],
        top_k: Optional[int] = None,
        focused_tables: Optional[Set[str]] = None,
    ) -> str:
        """Rank exemplars by vector cosine similarity to question and return formatted Markdown string.

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

        effective_k = top_k if top_k is not None else self.determine_adaptive_k(question, focused_tables)

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
        top_matches = scored[:effective_k]

        log.info(
            f"ExampleSelector: Selected top-{len(top_matches)} (adaptive_k={effective_k}) exemplars for '{question[:50]}' "
            f"(sim range: {top_matches[0][0]:.3f} to {top_matches[-1][0]:.3f})"
        )

        lines = [f"Relevant Query Examples (Top-{len(top_matches)}):"]
        for _, ex in top_matches:
            lines.append(f"Q: {ex['q']}")
            lines.append(f"A: {ex['a']}")
            lines.append("")

        return "\n".join(lines).strip()

    def clear_cache(self) -> None:
        """Clear cached exemplar vectors from memory."""
        self._exemplar_cache.clear()
        log.info("ExampleSelector: Exemplar vector cache cleared.")


# Singleton instance
example_selector = ExampleSelector()
