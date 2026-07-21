import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return arr
    return arr / norm


def cosine_similarity(a, b) -> float:
    """Cosine similarity; safe for already-normalized vectors."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))
