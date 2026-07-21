"""Simple IoU face tracker — keeps person ID when face briefly disappears."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    misses: int = 0
    hits: int = 0
    samples: list[tuple[float, np.ndarray]] = field(default_factory=list)
    votes: list[tuple[str, float]] = field(default_factory=list)
    label: str = "Unknown"
    score: float = 0.0
    confirmed: bool = False
    cycles: int = 0
    flushed: bool = False  # final recognize-on-leave done


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.25, max_misses: int = 45):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(
        self, faces: list[Any]
    ) -> tuple[list[tuple[Track, Any]], list[Track], list[Track]]:
        """
        Returns:
          matched  — (track, face) pairs this frame
          ghosts   — active tracks with no face this frame (still tracking)
          expiring — tracks being removed (face left scene)
        """
        det_boxes = [np.asarray(f.bbox, dtype=np.float32) for f in faces]
        track_ids = list(self.tracks.keys())
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        matched: list[tuple[Track, Any]] = []

        candidates = []
        for ti, tid in enumerate(track_ids):
            for di, box in enumerate(det_boxes):
                score = iou(self.tracks[tid].bbox, box)
                if score >= self.iou_threshold:
                    candidates.append((score, ti, di))
        candidates.sort(reverse=True)

        for score, ti, di in candidates:
            tid = track_ids[ti]
            if tid in assigned_tracks or di in assigned_dets:
                continue
            track = self.tracks[tid]
            track.bbox = det_boxes[di]
            track.misses = 0
            track.hits += 1
            assigned_tracks.add(tid)
            assigned_dets.add(di)
            matched.append((track, faces[di]))

        for di, face in enumerate(faces):
            if di in assigned_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            track = Track(track_id=tid, bbox=det_boxes[di], hits=1)
            self.tracks[tid] = track
            matched.append((track, face))

        ghosts: list[Track] = []
        expiring: list[Track] = []
        dead: list[int] = []

        for tid, track in self.tracks.items():
            if tid in assigned_tracks:
                continue
            track.misses += 1
            if track.misses > self.max_misses:
                dead.append(tid)
                expiring.append(track)
            else:
                ghosts.append(track)

        for tid in dead:
            del self.tracks[tid]

        return matched, ghosts, expiring
