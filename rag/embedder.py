"""
embedder.py — Converts text into a float vector for similarity search.

PRIMARY  : Ollama nomic-embed-text (real semantic embeddings).
FALLBACK : Character trigram + word unigram hash vectors.
           Zero external dependencies — switches automatically if Ollama
           is offline or the embedding model hasn't been pulled yet.
"""

import re
import math
import hashlib
import ollama

from config.settings import DEFAULT_EMBED_MODEL, MOCK_DIM
from config.logger import get_logger

log = get_logger(__name__)


class TextEmbedder:
    """
    Wraps Ollama's embedding API with an automatic trigram-based fallback.

    Usage:
        embedder = TextEmbedder()
        vector = embedder.embed("show me all employees")
    """

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
        self.embed_model       = embed_model
        self._ollama_available = False
        self._probe_ollama()

    def _probe_ollama(self) -> None:
        """Send a test call to Ollama to check if the embed model is available."""
        try:
            ollama.embeddings(model=self.embed_model, prompt="ping")
            self._ollama_available = True
            log.info(f"TextEmbedder: Ollama embedding active ({self.embed_model})")
        except Exception as e:
            self._ollama_available = False
            log.warning(
                f"TextEmbedder: Ollama embed unavailable "
                f"({type(e).__name__}: {e}). "
                f"Switching to Mock trigram mode — "
                f"to enable real embeddings run: ollama pull {self.embed_model}"
            )

    def embed(self, text: str) -> list[float]:
        """
        Embed text using Ollama if available, otherwise use mock trigram vectors.

        Args:
            text: Any plain-text string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if self._ollama_available:
            return self._ollama_embed(text)
        return self._mock_embed(text)

    def _ollama_embed(self, text: str) -> list[float]:
        """Call Ollama's embedding API. Falls back to mock on runtime failure."""
        try:
            resp = ollama.embeddings(model=self.embed_model, prompt=text)
            return resp["embedding"]
        except Exception as e:
            log.warning(
                f"TextEmbedder: Ollama embed runtime failure ({e}). "
                f"Switching to mock for this session."
            )
            self._ollama_available = False
            return self._mock_embed(text)

    def _mock_embed(self, text: str, dim: int = MOCK_DIM) -> list[float]:
        """
        Zero-dependency pseudo-embedding using character trigrams + word
        unigrams hashed into a fixed-dimension bucket vector.

        Similar words → overlapping trigrams → high cosine similarity.
        No numpy, no sklearn, no external packages required.
        """
        text = re.sub(r'\s+', ' ', text.lower().strip())
        vec  = [0.0] * dim

        # Word unigrams (higher weight — whole-word matches matter more)
        for word in text.split():
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dim
            vec[h] += 3.0

        # Character trigrams (capture morphological similarity)
        for i in range(len(text) - 2):
            trigram = text[i : i + 3]
            h = int(hashlib.sha256(trigram.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0

        # L2 normalize so cosine similarity is well-defined
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]
