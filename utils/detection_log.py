"""Thread-safe detailed detection log (per-frame names + confidence %)."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Iterable


class DetectionLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Face recognition detection log\n"
                f"# started={datetime.now().isoformat(timespec='seconds')}\n"
                f"# columns: timestamp | camera | frame | face details + confidence %\n"
                f"{'=' * 80}\n"
            )
        print(f"Detection log -> {self.log_path}")

    def log_frame(
        self,
        camera_id: str,
        camera_name: str,
        frame_id: int,
        detections: Iterable[dict],
    ) -> None:
        """
        detections items:
          {
            "label": "ali"|"Customer",
            "score": 0.87,
            "scores": {"ali": 0.87, "ritesh": 0.41},
            "bbox": (x1,y1,x2,y2) optional
          }
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        detections = list(detections)

        lines = [
            f"{ts} | camera={camera_id} ({camera_name}) | frame={frame_id} "
            f"| faces={len(detections)}"
        ]

        if not detections:
            lines.append("  no_faces")
        else:
            for idx, det in enumerate(detections, start=1):
                label = det["label"]
                score = float(det["score"])
                pct = score * 100.0
                score_parts = []
                for name, value in sorted(
                    (det.get("scores") or {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    score_parts.append(f"{name}={value * 100:.2f}%")
                scores_txt = ", ".join(score_parts) if score_parts else "n/a"
                bbox = det.get("bbox")
                bbox_txt = ""
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    bbox_txt = f" | bbox=({x1},{y1},{x2},{y2})"
                extras = []
                if det.get("track_id") is not None:
                    extras.append(f"track={det['track_id']}")
                if "confirmed" in det:
                    extras.append(f"confirmed={det['confirmed']}")
                if "samples" in det:
                    extras.append(f"samples={det['samples']}")
                if "votes" in det:
                    extras.append(f"votes={det['votes']}")
                extra_txt = (" | " + " | ".join(extras)) if extras else ""
                lines.append(
                    f"  face#{idx} | label={label} | confidence={pct:.2f}% "
                    f"| all_scores: {scores_txt}{bbox_txt}{extra_txt}"
                )

        text = "\n".join(lines) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
