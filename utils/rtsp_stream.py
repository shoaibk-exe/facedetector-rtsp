"""Robust RTSP open/decode helpers: skip corrupt frames, reconnect only when needed."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from utils.config import (
    RTSP_BUFFER_SIZE,
    RTSP_FFMPEG_OPTIONS,
    RTSP_RECONNECT_DELAY,
    RTSP_SKIP_BEFORE_RECONNECT,
)


def configure_ffmpeg_for_rtsp() -> None:
    """
    Must run before VideoCapture. Tells FFmpeg to prefer TCP, low latency,
    and discard corrupt H.264 access units (fixes 'missing picture in access unit').
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_FFMPEG_OPTIONS
    # Quiet most FFmpeg spam (8 = AV_LOG_FATAL in many builds; 16 = ERROR)
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def open_rtsp(url: str) -> cv2.VideoCapture:
    configure_ffmpeg_for_rtsp()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        # One retry after a short pause (NVR sometimes rejects burst connects)
        time.sleep(0.5)
        cap.release()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {url}")

    # Keep only the newest encoded packet — reduces backlog/decode errors
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
    except Exception:
        pass

    # Soft timeouts when supported by the build
    for prop, value in (
        (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), 10000),
        (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None), 8000),
    ):
        if prop is not None:
            try:
                cap.set(prop, value)
            except Exception:
                pass

    return cap


def is_valid_frame(frame) -> bool:
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if frame.ndim < 2:
        return False
    h, w = frame.shape[:2]
    return h >= 16 and w >= 16


def read_good_frame(cap: cv2.VideoCapture):
    """
    Read one frame. Returns (True, frame) on success.
    On corrupt/missing decode, returns (False, None) without raising.
    """
    try:
        # grab() pulls packet; retrieve() decodes — split so we can drop bad packets
        if not cap.grab():
            return False, None
        ok, frame = cap.retrieve()
        if not ok or not is_valid_frame(frame):
            return False, None
        return True, frame
    except cv2.error:
        return False, None
    except Exception:
        return False, None


class RtspFrameGrabber(threading.Thread):
    """
    Continuously grabs RTSP frames in a background thread.
    Corrupt / missing frames are skipped. Reconnects only after many
    consecutive failures (not on every glitch).
    Always keeps the *latest* good frame for the detector.
    """

    def __init__(self, url: str, cam_id: str, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"grab-{cam_id}")
        self.url = url
        self.cam_id = cam_id
        self.stop_event = stop_event

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self.skipped = 0
        self.reconnects = 0
        self.error: str | None = None

    def get_latest(self, copy: bool = True):
        with self._lock:
            if self._latest is None:
                return None, self._seq
            frame = self._latest.copy() if copy else self._latest
            return frame, self._seq

    def _store(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self._seq += 1

    def _reconnect(self, cap: Optional[cv2.VideoCapture]) -> cv2.VideoCapture:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        self.reconnects += 1
        print(
            f"[{self.cam_id}] Reconnecting RTSP "
            f"(attempt {self.reconnects}, skipped={self.skipped})..."
        )
        time.sleep(RTSP_RECONNECT_DELAY)
        return open_rtsp(self.url)

    def run(self) -> None:
        cap = None
        consecutive_bad = 0

        try:
            cap = open_rtsp(self.url)
            print(f"[{self.cam_id}] RTSP grabber ready")

            while not self.stop_event.is_set():
                ok, frame = read_good_frame(cap)

                if ok:
                    consecutive_bad = 0
                    self._store(frame)
                    continue

                # Corrupt / missing access unit / empty grab — skip, do not reconnect yet
                consecutive_bad += 1
                self.skipped += 1

                if consecutive_bad == 1 or consecutive_bad % 30 == 0:
                    print(
                        f"[{self.cam_id}] Skipping bad/missing frame "
                        f"(streak={consecutive_bad}, total_skipped={self.skipped})"
                    )

                if consecutive_bad >= RTSP_SKIP_BEFORE_RECONNECT:
                    try:
                        cap = self._reconnect(cap)
                        consecutive_bad = 0
                    except Exception as exc:
                        self.error = str(exc)
                        print(f"[{self.cam_id}] Reconnect failed: {exc}")
                        time.sleep(RTSP_RECONNECT_DELAY)
                else:
                    # Brief yield so we don't spin the CPU on a dead socket
                    time.sleep(0.01)

        except Exception as exc:
            self.error = str(exc)
            print(f"[{self.cam_id}] Grabber crashed: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            print(f"[{self.cam_id}] Grabber stopped")
