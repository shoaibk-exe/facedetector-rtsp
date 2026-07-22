"""
Simple recognition pipeline:

detect face → embedding → best staff match → show name + score
No quality filter, tracking collect, or vote confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils.gallery import Gallery, match_max
from utils.quality import get_normed_embedding


@dataclass
class DisplayFace:
    bbox: tuple
    label: str
    score: float
    track_id: int = 0
    confirmed: bool = True
    quality_ok: bool = True
    samples: int = 1
    votes: int = 1
    ghost: bool = False
    scores: dict | None = None


class RecognitionPipeline:
    def __init__(self, gallery: Gallery, threshold: float = 0.0):
        self.gallery = gallery
        # threshold kept for API compatibility; matching always returns best staff
        self.threshold = threshold

    def process(self, frame_bgr, faces) -> List[DisplayFace]:
        display: List[DisplayFace] = []

        for idx, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            emb = get_normed_embedding(face)

            if emb is None:
                display.append(
                    DisplayFace(
                        bbox=(x1, y1, x2, y2),
                        label="Unknown",
                        score=0.0,
                        track_id=idx,
                        confirmed=False,
                        scores={},
                    )
                )
                continue

            # Always show best staff guess (no quality / collect / vote gates)
            label, score, scores = match_max(emb, self.gallery, threshold=-1.0)
            display.append(
                DisplayFace(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    score=score,
                    track_id=idx,
                    confirmed=True,
                    scores=scores,
                )
            )

        return display

    def staff_names_confirmed(self, display: List[DisplayFace]) -> List[str]:
        names = []
        for d in display:
            if d.label in {"Unknown", ""}:
                continue
            names.append(d.label)
        return names
