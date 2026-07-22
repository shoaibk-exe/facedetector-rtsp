"""
Production recognition pipeline:

detect → quality filter → track → collect frames → top-K → average
→ max-similarity match → vote → confirm

Recognizes with fewer frames when the person leaves before full collection.
Shows track box even when face is briefly missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from utils.config import (
    COLLECT_FRAMES,
    CONFIRM_VOTES,
    MIN_COLLECT_FRAMES,
    TOP_QUALITY_FRAMES,
    TRACK_IOU,
    TRACK_MAX_MISSES,
)
from utils.gallery import Gallery, average_embeddings, match_max
from utils.quality import get_normed_embedding, score_face
from utils.tracker import FaceTracker, Track


@dataclass
class DisplayFace:
    bbox: tuple
    label: str
    score: float
    track_id: int
    confirmed: bool
    quality_ok: bool
    samples: int
    votes: int
    ghost: bool = False  # face not seen this frame, box from last position


class RecognitionPipeline:
    def __init__(
        self,
        gallery: Gallery,
        threshold: float,
        collect_frames: int = COLLECT_FRAMES,
        min_collect_frames: int = MIN_COLLECT_FRAMES,
        top_k: int = TOP_QUALITY_FRAMES,
        confirm_votes: int = CONFIRM_VOTES,
    ):
        self.gallery = gallery
        self.threshold = threshold
        self.collect_frames = max(1, collect_frames)
        self.min_collect_frames = max(1, min(min_collect_frames, collect_frames))
        self.top_k = max(1, min(top_k, self.collect_frames))
        self.confirm_votes = max(1, confirm_votes)
        self.tracker = FaceTracker(
            iou_threshold=TRACK_IOU,
            max_misses=TRACK_MAX_MISSES,
        )

    def _run_match(self, track: Track, sample_count: int | None = None) -> None:
        n = sample_count or len(track.samples)
        if n < self.min_collect_frames:
            return

        ranked = sorted(track.samples, key=lambda item: item[0], reverse=True)
        k = min(self.top_k, len(ranked), n)
        top = ranked[:k]
        emb = average_embeddings([e for _, e in top])
        label, score, _ = match_max(emb, self.gallery, self.threshold)

        track.votes.append((label, score))
        track.cycles += 1
        keep = max(self.min_collect_frames, self.collect_frames // 2)
        track.samples = track.samples[-keep:]
        self._update_confirmation(track)

    def _recognize_track(self, track: Track, force: bool = False) -> None:
        """Full collection cycle, or partial if force=True (person leaving)."""
        if force:
            if len(track.samples) >= self.min_collect_frames and not track.flushed:
                self._run_match(track)
                track.flushed = True
            return

        if len(track.samples) >= self.collect_frames:
            self._run_match(track)

    def _flush_expiring(self, tracks: List[Track]) -> None:
        for track in tracks:
            self._recognize_track(track, force=True)

    def _update_confirmation(self, track: Track) -> None:
        if not track.votes:
            track.label = "Unknown"
            track.score = 0.0
            track.confirmed = False
            return

        recent = track.votes[-self.confirm_votes :]
        names = [n for n, _ in recent]
        if (
            len(recent) >= self.confirm_votes
            and len(set(names)) == 1
            and names[0] != "Unknown"
        ):
            track.label = names[0]
            track.score = float(np.mean([s for _, s in recent]))
            track.confirmed = True
            return

        counts: Dict[str, list] = {}
        for name, score in track.votes:
            counts.setdefault(name, []).append(score)
        best = max(
            counts.keys(),
            key=lambda n: (len(counts[n]), float(np.mean(counts[n]))),
        )
        track.label = best
        track.score = float(np.mean(counts[best]))
        track.confirmed = False

    def _display_label(self, track: Track, quality_ok: bool, quality_score: float):
        if track.confirmed:
            return track.label, track.score, True

        target = self.collect_frames
        if len(track.samples) < target and not track.votes:
            return (
                f"track {len(track.samples)}/{target}",
                quality_score if quality_ok else 0.0,
                False,
            )

        if track.votes:
            suffix = "?" if track.label != "Unknown" else ""
            return f"{track.label}{suffix}", track.score, False

        return "Unknown", 0.0, False

    def _track_to_display(
        self, track: Track, bbox, quality_ok: bool, quality_score: float, ghost: bool
    ) -> DisplayFace:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label, score, confirmed = self._display_label(
            track, quality_ok, quality_score
        )
        if ghost and not confirmed:
            label = f"{label} (no face)"

        return DisplayFace(
            bbox=(x1, y1, x2, y2),
            label=label,
            score=score,
            track_id=track.track_id,
            confirmed=confirmed,
            quality_ok=quality_ok,
            samples=len(track.samples),
            votes=len(track.votes),
            ghost=ghost,
        )

    def process(self, frame_bgr, faces) -> List[DisplayFace]:
        matched, ghosts, expiring = self.tracker.update(faces)
        self._flush_expiring(expiring)

        display: List[DisplayFace] = []

        for track, face in matched:
            quality = score_face(frame_bgr, face)
            if quality.ok:
                emb = get_normed_embedding(face)
                if emb is not None:
                    track.samples.append((quality.score, emb))
                    self._recognize_track(track)

            display.append(
                self._track_to_display(
                    track, face.bbox, quality.ok, quality.score, ghost=False
                )
            )

        for track in ghosts:
            display.append(
                self._track_to_display(
                    track, track.bbox, False, 0.0, ghost=True
                )
            )

        return display

    def staff_names_confirmed(self, display: List[DisplayFace]) -> List[str]:
        names = []
        for d in display:
            if not d.confirmed or d.ghost:
                continue
            if d.label in {"Unknown", ""}:
                continue
            if "track " in d.label or d.label.endswith("?"):
                continue
            if "(no face)" in d.label:
                continue
            names.append(d.label.split(" (")[0])
        return names
