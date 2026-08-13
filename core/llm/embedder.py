"""
embedder.py — Converts text into a float vector for similarity search.

PRIMARY  : Local SentenceTransformer model (all-MiniLM-L6-v2).
FALLBACK : Character trigram + word unigram hash vectors.
           Zero external server dependencies — 100% offline execution.
"""

# ── MODULE TAG: Text Embedder Service ──
import re
import math
import hashlib
from functools import lru_cache

from core.config.settings import DEFAULT_EMBED_MODEL, MOCK_DIM
from core.config.logger import get_logger

log = get_logger(__name__)

_st_model = None
_st_failed = False


def _get_st_model():
    """Load the SentenceTransformer model when needed.

    Returns:
        SentenceTransformer | None: The loaded AI embedding model, or None if loading failed.
    """
    global _st_model, _st_failed
    if _st_failed:
        return None
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("TextEmbedder: Loaded local SentenceTransformer model 'all-MiniLM-L6-v2'.")
        except Exception as e:
            log.warning(f"TextEmbedder: SentenceTransformer load failed ({e}). Using mock trigram vectors.")
            _st_failed = True
            return None
    return _st_model


class TextEmbedder:
    """Converts text into a list of numbers (embeddings) for quick similarity search.

    Attributes:
        embed_model (str): Name of the embedding model being used.
        _api_available (bool): True if the main model is working, False if using fallback mode.
    """

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
        """Set up the TextEmbedder instance.

        Args:
            embed_model (str, optional): Name of the embedding model. Defaults to DEFAULT_EMBED_MODEL.
        """
        self.embed_model = embed_model
        self._api_available = True

    @property
    def _ollama_available(self) -> bool:
        return self._api_available

    @_ollama_available.setter
    def _ollama_available(self, val: bool) -> None:
        self._api_available = val

    @lru_cache(maxsize=1024)
    def _embed_cached_tuple(self, text: str) -> tuple[float, ...]:
        """Internal helper that remembers recent calculations so we don't repeat work.

        Args:
            text (str): The text to convert into numbers.

        Returns:
            tuple[float, ...]: A tuple of numbers representing the meaning of the text.
        """
        if self._api_available:
            vec = self._local_embed(text)
        else:
            vec = self._mock_embed(text)
        return tuple(vec)

    def embed(self, text: str) -> list[float]:
        """Convert text into a list of numbers representing its meaning.

        Args:
            text (str): The input text to convert.

        Returns:
            list[float]: A list of floating-point numbers.
        """
        return list(self._embed_cached_tuple(text))

    def clear_cache(self) -> None:
        """Wipe the saved embedding calculations from memory."""
        self._embed_cached_tuple.cache_clear()
        log.info("TextEmbedder: In-memory embedding LRU cache cleared.")

    def _local_embed(self, text: str) -> list[float]:
        """Convert text to numbers using the local SentenceTransformer model.

        Args:
            text (str): The input text.

        Returns:
            list[float]: A list of numbers representing the text.
        """
        model = _get_st_model()
        if model is not None:
            try:
                vec = model.encode(text, normalize_embeddings=True)
                return [float(x) for x in vec]
            except Exception as e:
                log.warning(f"TextEmbedder: Local embedding computation failed ({e}). Falling back to mock.")
        return self._mock_embed(text)

    def _mock_embed(self, text: str, dim: int = MOCK_DIM) -> list[float]:
        """Simple fallback method to convert text to numbers without external dependencies.

        Args:
            text (str): The input text.
            dim (int, optional): Size of the output number list. Defaults to MOCK_DIM.

        Returns:
            list[float]: A simplified list of numbers representing words and letter groups.
        """
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
