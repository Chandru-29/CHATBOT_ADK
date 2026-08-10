"""
embedder.py — Converts text into a float vector for similarity search.

PRIMARY  : Google Gemini API text-embedding-004 via google-genai SDK.
FALLBACK : Character trigram + word unigram hash vectors.
           Zero external server dependencies — switches automatically if
           offline or unauthenticated.
"""

# ── MODULE TAG: Text Embedder Service ──
import re
import math
import hashlib
from functools import lru_cache

from core.config.settings import GEMINI_EMBED_MODEL, MOCK_DIM
from core.config.logger import get_logger
from core.llm.llm_client import get_genai_client

log = get_logger(__name__)


class TextEmbedder:
    """
    Wraps Google Gemini embedding API with automatic trigram-based fallback
    and in-memory LRU caching to avoid redundant network embedding calls.
    """

    def __init__(self, embed_model: str = GEMINI_EMBED_MODEL) -> None:
        self.embed_model = embed_model
        self._api_available = True

    @property
    def _ollama_available(self) -> bool:
        return self._api_available

    @_ollama_available.setter
    def _ollama_available(self, val: bool) -> None:
        self._api_available = val

    @lru_cache(maxsize=512)
    def _embed_cached_tuple(self, text: str) -> tuple[float, ...]:
        """Internal cached helper returning immutable float tuple."""
        if self._api_available:
            vec = self._gemini_embed(text)
        else:
            vec = self._mock_embed(text)
        return tuple(vec)

    def embed(self, text: str) -> list[float]:
        """Embed text using Gemini API if available, otherwise use mock trigram vectors."""
        return list(self._embed_cached_tuple(text))

    def clear_cache(self) -> None:
        """Clear the in-memory embedding LRU cache."""
        self._embed_cached_tuple.cache_clear()
        log.info("TextEmbedder: In-memory embedding LRU cache cleared.")

    def _gemini_embed(self, text: str) -> list[float]:
        """Call Google Gemini's embedding API. Falls back to mock on runtime failure."""
        try:
            client = get_genai_client()
            res = client.models.embed_content(
                model=self.embed_model,
                contents=text,
            )
            if hasattr(res, "embedding") and res.embedding and hasattr(res.embedding, "values"):
                return list(res.embedding.values)
            elif hasattr(res, "embeddings") and res.embeddings and len(res.embeddings) > 0 and hasattr(res.embeddings[0], "values"):
                return list(res.embeddings[0].values)
            return self._mock_embed(text)
        except Exception as e:
            log.warning(f"TextEmbedder: Gemini embedding call failed ({e}). Switching to mock for this session.")
            self._api_available = False
            return self._mock_embed(text)

    def _mock_embed(self, text: str, dim: int = MOCK_DIM) -> list[float]:
        """Zero-dependency pseudo-embedding using character trigrams + word unigrams."""
        text = re.sub(r'\s+', ' ', text.lower().strip())
        vec = [0.0] * dim

        for word in text.split():
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim
            vec[h] += 3.0

        for i in range(len(text) - 2):
            trigram = text[i : i + 3]
            h = int(hashlib.sha256(trigram.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0

        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]
