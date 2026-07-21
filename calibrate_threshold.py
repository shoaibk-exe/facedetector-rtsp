"""
Calibrate recognition threshold on YOUR CCTV gallery.

Builds genuine pairs (same person, different embeddings) and impostor pairs
(different people), prints score ranges, and suggests a threshold.

Usage:
  python calibrate_threshold.py
  python calibrate_threshold.py --database database/embeddings.pkl
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from utils.config import DATABASE_PATH, THRESHOLD
from utils.gallery import load_gallery


def pair_scores(gallery: dict) -> tuple[list[float], list[float]]:
    genuine: list[float] = []
    impostor: list[float] = []

    names = list(gallery.keys())
    for name, embs in gallery.items():
        n = embs.shape[0]
        for i, j in itertools.combinations(range(n), 2):
            genuine.append(float(np.dot(embs[i], embs[j])))

    for a, b in itertools.combinations(names, 2):
        ea, eb = gallery[a], gallery[b]
        # sample up to 200 impostor pairs per identity pair
        sims = ea @ eb.T
        flat = sims.reshape(-1)
        if flat.size > 200:
            rng = np.random.default_rng(42)
            flat = rng.choice(flat, size=200, replace=False)
        impostor.extend(float(x) for x in flat)

    return genuine, impostor


def suggest_threshold(genuine: list[float], impostor: list[float]) -> float:
    if not genuine or not impostor:
        return THRESHOLD

    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)

    # Sweep thresholds; pick max (TPR - FPR) Youden-like on embedding pairs
    candidates = np.linspace(0.15, 0.70, 56)
    best_t, best_score = THRESHOLD, -1.0
    for t in candidates:
        tpr = float(np.mean(g >= t))
        fpr = float(np.mean(i >= t))
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def main():
    parser = argparse.ArgumentParser(description="Calibrate face match threshold")
    parser.add_argument("--database", default=DATABASE_PATH)
    args = parser.parse_args()

    gallery = load_gallery(args.database)
    print(f"Loaded {len(gallery)} identities from {args.database}")
    for name, embs in gallery.items():
        print(f"  {name}: {embs.shape[0]} embedding(s)")

    genuine, impostor = pair_scores(gallery)
    print(f"\nGenuine pairs: {len(genuine)}")
    print(f"Impostor pairs: {len(impostor)}")

    if genuine:
        g = np.asarray(genuine)
        print(
            f"Genuine  min/mean/max: {g.min():.3f} / {g.mean():.3f} / {g.max():.3f}"
        )
    if impostor:
        i = np.asarray(impostor)
        print(
            f"Impostor min/mean/max: {i.min():.3f} / {i.mean():.3f} / {i.max():.3f}"
        )

    suggested = suggest_threshold(genuine, impostor)
    print(f"\nCurrent THRESHOLD in .env: {THRESHOLD}")
    print(f"Suggested THRESHOLD:        {suggested:.2f}")
    print("Set THRESHOLD in .env after validating on held-out CCTV clips.")
    print("Below threshold → label is Unknown (not forced to a staff name).")


if __name__ == "__main__":
    main()
