"""
similarity.py — Pure-Python cosine similarity calculation.

No numpy or external packages required. Used by schema_retriever.py to
score how closely a user's question vector matches each table's vector.
"""

import math


def cosine_sim(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length float vectors.

    Returns a value between -1.0 and 1.0.
    Returns 0.0 if either vector is a zero vector (avoids division by zero).

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score as a float.
    """
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
