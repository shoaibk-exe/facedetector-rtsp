"""Save short video clips when staff faces are detected."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Optional

import cv2
import numpy as np


class StaffClipRecorder:
    """
    Starts an MP4 when staff is seen; keeps writing while staff is present;
    closes after `post_pad_frames` frames with no staff.
    """

    def __init__(
        self,
        camera_id: str,
        output_dir: str,
        fps: float = 15.0,
        post_pad_frames: int = 45,
    ):
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.fps = fps
        self.post_pad_frames = max(1, post_pad_frames)

        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[str] = None
        self._names: set[str] = set()
        self._idle = 0
        self._size: Optional[tuple[int, int]] = None
        self.clips_saved = 0

        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def active(self) -> bool:
        return self._writer is not None

    def _start(self, frame: np.ndarray, staff_names: Iterable[str]) -> None:
        h, w = frame.shape[:2]
        self._size = (w, h)
        self._names = {n for n in staff_names if n}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        people = "_".join(sorted(self._names)) or "staff"
        safe_people = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in people
        )
        self._path = os.path.join(
            self.output_dir,
            f"{self.camera_id}_{safe_people}_{stamp}.mp4",
        )
        self._writer = cv2.VideoWriter(
            self._path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )
        self._idle = 0
        print(f"[{self.camera_id}] Staff clip START -> {self._path}")

    def _stop(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        self.clips_saved += 1
        names = ",".join(sorted(self._names)) or "staff"
        print(
            f"[{self.camera_id}] Staff clip SAVED ({names}) -> {self._path}"
        )
        self._path = None
        self._names.clear()
        self._idle = 0

    def update(self, frame: np.ndarray, staff_names: Iterable[str]) -> None:
        names = [n for n in staff_names if n and n != "Customer"]

        if names:
            if self._writer is None:
                self._start(frame, names)
            else:
                self._names.update(names)
            self._idle = 0
            self._writer.write(frame)
            return

        if self._writer is None:
            return

        # Staff left — keep a short tail, then close
        self._writer.write(frame)
        self._idle += 1
        if self._idle >= self.post_pad_frames:
            self._stop()

    def close(self) -> None:
        self._stop()
