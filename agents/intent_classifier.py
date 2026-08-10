"""
intent_classifier.py — Hugging Face all-MiniLM-L6-v2 Zero-LLM Embedding Intent Classifier.

Uses sentence-transformers / all-MiniLM-L6-v2 embeddings locally to classify user intent:
  - WMS_AGENT  : Database data intent (picklists, items, inventory, GRN, locations, etc.)
  - GENERAL    : Greetings, thanks, farewells, off-topic small talk
"""

# ── MODULE TAG: Zero-LLM Embedding Intent Classifier ──
import numpy as np

from core.config.logger import get_logger
from prompts.loader import prompt_loader

log = get_logger(__name__)

_classifier_instance = None


class EmbeddingIntentClassifier:
    """Zero-LLM Intent Classifier using local sentence-transformers all-MiniLM-L6-v2 embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.exemplar_vectors = {}
        self._fallback_mode = False
        self._init_classifier()

    def _init_classifier(self):
        """Initialize SentenceTransformer model and precompute exemplar embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"IntentClassifier: Loading local HuggingFace embedding model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self._precompute_exemplars()
            log.info("IntentClassifier: SentenceTransformer model & exemplar embeddings initialized successfully.")
        except Exception as e:
            log.warning(f"IntentClassifier: SentenceTransformer load failed ({e}). Falling back to TextEmbedder.")
            self._fallback_mode = True

    def _precompute_exemplars(self):
        """Extract exemplars from examples.yml via PromptLoader and pre-compute embedding vectors."""
        exemplar_dict = prompt_loader.get_exemplar_questions_by_intent()
        for intent, questions in exemplar_dict.items():
            if questions:
                embeddings = self.model.encode(questions, normalize_embeddings=True)
                self.exemplar_vectors[intent] = {
                    "questions": questions,
                    "embeddings": embeddings,
                }
                log.debug(f"IntentClassifier: Embedded {len(questions)} exemplars for domain '{intent}'")

    def predict(self, question: str) -> str:
        """Classify question intent by computing cosine similarity against pre-computed exemplar vectors."""
        if self._fallback_mode or not self.model:
            return self._fallback_predict(question)

        try:
            q_vec = self.model.encode([question], normalize_embeddings=True)[0]
            best_intent = "WMS_AGENT"
            best_score = -1.0

            for intent, data in self.exemplar_vectors.items():
                matrix = data["embeddings"]
                sims = np.dot(matrix, q_vec)
                max_sim = float(np.max(sims))
                if max_sim > best_score:
                    best_score = max_sim
                    best_intent = intent

            log.info(f"IntentClassifier: query='{question[:50]}' -> intent={best_intent} (top sim score={best_score:.4f})")
            return best_intent
        except Exception as e:
            log.warning(f"IntentClassifier: Embedding inference error ({e}). Falling back to WMS_AGENT.")
            return "WMS_AGENT"

    def predict_with_score(self, question: str) -> tuple[str, float]:
        """Classify question intent and return (intent, top_cosine_similarity_score)."""
        if self._fallback_mode or not self.model:
            return self._fallback_predict(question), 0.88

        try:
            q_vec = self.model.encode([question], normalize_embeddings=True)[0]
            best_intent = "WMS_AGENT"
            best_score = 0.0

            for intent, data in self.exemplar_vectors.items():
                matrix = data["embeddings"]
                sims = np.dot(matrix, q_vec)
                max_sim = float(np.max(sims))
                if max_sim > best_score:
                    best_score = max_sim
                    best_intent = intent

            return best_intent, best_score
        except Exception as e:
            log.warning(f"IntentClassifier: Embedding inference error ({e}). Falling back to WMS_AGENT.")
            return "WMS_AGENT", 0.88

    def _fallback_predict(self, question: str) -> str:
        """Fallback prediction using TextEmbedder cosine similarity."""
        from core.llm.embedder import TextEmbedder
        embedder = TextEmbedder()
        q_vec = embedder.embed(question)
        exemplar_dict = prompt_loader.get_exemplar_questions_by_intent()

        best_intent = "WMS_AGENT"
        best_score = -1.0

        for intent, questions in exemplar_dict.items():
            for q in questions:
                v = embedder.embed(q)
                dot = sum(a * b for a, b in zip(q_vec, v))
                if dot > best_score:
                    best_score = dot
                    best_intent = intent

        return best_intent


def get_intent_classifier() -> EmbeddingIntentClassifier:
    """Return singleton instance of EmbeddingIntentClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmbeddingIntentClassifier()
    return _classifier_instance


def classify_intent(question: str) -> str:
    """Classify user question intent using HuggingFace all-MiniLM-L6-v2 embeddings."""
    clf = get_intent_classifier()
    return clf.predict(question)
