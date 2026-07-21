"""Multi-embedding staff gallery with max-similarity matching."""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from utils.similarity import cosine_similarity, l2_normalize


Gallery = Dict[str, np.ndarray]  # name -> (N, D) float32 normalized


def _as_matrix(value) -> np.ndarray:
    """Accept legacy single vector or list/array of embeddings."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    # Normalize each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return arr / norms


def load_gallery(path: str) -> Gallery:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Staff database not found: {path}\n"
            "Run enroll_staff.py first."
        )
    with open(path, "rb") as f:
        raw = pickle.load(f)

    gallery: Gallery = {}
    for name, value in raw.items():
        mat = _as_matrix(value)
        if mat.size == 0:
            continue
        gallery[str(name)] = mat
    return gallery


def save_gallery(path: str, gallery: Gallery) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {name: np.asarray(embs, dtype=np.float32) for name, embs in gallery.items()}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def match_max(
    embedding: np.ndarray,
    gallery: Gallery,
    threshold: float,
) -> Tuple[str, float, Dict[str, float]]:
    """
    For each identity, score = max cosine over that person's embeddings.
    Return Unknown if best < threshold.
    """
    query = l2_normalize(embedding)
    per_person: Dict[str, float] = {}

    for name, embs in gallery.items():
        # embs already normalized; query · emb.T
        sims = embs @ query
        per_person[name] = float(np.max(sims))

    if not per_person:
        return "Unknown", -1.0, {}

    best_name = max(per_person, key=per_person.get)
    best_score = per_person[best_name]
    if best_score < threshold:
        return "Unknown", best_score, per_person
    return best_name, best_score, per_person


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack([l2_normalize(e) for e in embeddings], axis=0)
    mean = stacked.mean(axis=0)
    return l2_normalize(mean)
