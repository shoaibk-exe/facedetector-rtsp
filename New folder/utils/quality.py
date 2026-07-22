"""Face quality scoring: size, blur, pose. Reject bad frames before embedding use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from utils.config import (
    MIN_FACE_WIDTH,
    MIN_BLUR_SCORE,
    MAX_YAW_DEG,
    MAX_PITCH_DEG,
    MAX_ROLL_DEG,
)


@dataclass
class QualityResult:
    ok: bool
    score: float
    reason: str = ""
    face_width: float = 0.0
    blur: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


def _face_width(bbox) -> float:
    x1, _, x2, _ = bbox
    return float(max(0.0, x2 - x1))


def blur_score(frame_bgr: np.ndarray, bbox) -> float:
    """Laplacian variance on the face crop — higher = sharper."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_pose_deg(face) -> tuple[float, float, float]:
    """
    Return (pitch, yaw, roll) in degrees.
    Prefer InsightFace face.pose; fall back to 5-point landmark heuristic.
    """
    pose = getattr(face, "pose", None)
    if pose is not None:
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return float(arr[0]), float(arr[1]), float(arr[2])

    kps = getattr(face, "kps", None)
    if kps is None:
        return 0.0, 0.0, 0.0

    pts = np.asarray(kps, dtype=np.float64)
    if pts.shape[0] < 5:
        return 0.0, 0.0, 0.0

    # InsightFace 5-point order: left_eye, right_eye, nose, left_mouth, right_mouth
    left_eye, right_eye, nose = pts[0], pts[1], pts[2]
    eye_mid = (left_eye + right_eye) * 0.5
    eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6

    # Yaw proxy: nose offset from eye midpoint along x
    yaw = float(np.clip((nose[0] - eye_mid[0]) / eye_dist * 90.0, -90.0, 90.0))
    # Pitch proxy: nose vs eye midline along y
    pitch = float(np.clip((nose[1] - eye_mid[1]) / eye_dist * 90.0, -90.0, 90.0))
    # Roll: eye line angle
    roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    return pitch, yaw, roll


def score_face(
    frame_bgr: np.ndarray,
    face,
    min_face_width: float = MIN_FACE_WIDTH,
    min_blur: float = MIN_BLUR_SCORE,
    max_yaw: float = MAX_YAW_DEG,
    max_pitch: float = MAX_PITCH_DEG,
    max_roll: float = MAX_ROLL_DEG,
) -> QualityResult:
    width = _face_width(face.bbox)
    if width < min_face_width:
        return QualityResult(
            ok=False,
            score=0.0,
            reason=f"face_width<{min_face_width}",
            face_width=width,
        )

    blur = blur_score(frame_bgr, face.bbox)
    if blur < min_blur:
        return QualityResult(
            ok=False,
            score=0.0,
            reason=f"blur<{min_blur}",
            face_width=width,
            blur=blur,
        )

    pitch, yaw, roll = estimate_pose_deg(face)
    if abs(yaw) > max_yaw:
        return QualityResult(
            ok=False,
            score=0.0,
            reason=f"yaw>{max_yaw}",
            face_width=width,
            blur=blur,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
    if abs(pitch) > max_pitch:
        return QualityResult(
            ok=False,
            score=0.0,
            reason=f"pitch>{max_pitch}",
            face_width=width,
            blur=blur,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )
    if abs(roll) > max_roll:
        return QualityResult(
            ok=False,
            score=0.0,
            reason=f"roll>{max_roll}",
            face_width=width,
            blur=blur,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    det = float(getattr(face, "det_score", 0.5) or 0.5)
    # Higher is better: sharpness + size + detection confidence − pose cost
    pose_cost = (abs(yaw) / max_yaw + abs(pitch) / max_pitch + abs(roll) / max_roll) / 3.0
    quality = (0.45 * min(blur / 200.0, 1.0)) + (0.25 * min(width / 160.0, 1.0))
    quality += 0.20 * det + 0.10 * (1.0 - min(pose_cost, 1.0))

    return QualityResult(
        ok=True,
        score=float(quality),
        reason="ok",
        face_width=width,
        blur=blur,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def get_normed_embedding(face) -> Optional[np.ndarray]:
    """Prefer InsightFace normed_embedding; fall back to L2-normalized embedding."""
    emb = getattr(face, "normed_embedding", None)
    if emb is not None:
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        return vec

    raw = getattr(face, "embedding", None)
    if raw is None:
        return None
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm
