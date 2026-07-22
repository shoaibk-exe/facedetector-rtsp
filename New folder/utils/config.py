import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_bool(key: str, default: bool = False) -> bool:
    value = _get(key, "1" if default else "0").lower()
    return value in {"1", "true", "yes", "on"}


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if not raw:
        return default
    return float(raw)


def build_rtsp_url() -> str:
    """Build RTSP URL from RTSP_URL or username/password/host parts."""
    full = _get("RTSP_URL")
    if full:
        return full

    username = _get("RTSP_USERNAME")
    password = _get("RTSP_PASSWORD")
    host = _get("RTSP_HOST")
    port = _get("RTSP_PORT", "554")
    path = _get("RTSP_PATH", "/")

    if not host:
        raise ValueError(
            "RTSP not configured. Set RTSP_URL or RTSP_HOST in .env"
        )

    if not path.startswith("/"):
        path = "/" + path

    userinfo = ""
    if username:
        safe_user = quote(username, safe="")
        safe_pass = quote(password, safe="")
        userinfo = f"{safe_user}:{safe_pass}@"

    return f"rtsp://{userinfo}{host}:{port}{path}"


def onnx_providers() -> list[str]:
    raw = _get("ONNX_PROVIDERS", "cuda,cpu").lower()
    mapping = {
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    providers = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        providers.append(mapping.get(part, part))
    return providers or ["CPUExecutionProvider"]


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if not raw:
        return default
    return int(raw)


SOURCE = _get("SOURCE", "mp4").lower()
THRESHOLD = _get_float("THRESHOLD", 0.35)
INPUT_DIR = _get("INPUT_DIR", "input")
OUTPUT_DIR = _get("OUTPUT_DIR", "output")
DATABASE_PATH = _get("DATABASE_PATH", "database/embeddings.pkl")
STAFF_DIR = _get("STAFF_DIR", "staff")
CAMERAS_FILE = _get("CAMERAS_FILE", "cameras.yaml")
SHOW_PREVIEW = _get_bool("SHOW_PREVIEW", True)
SAVE_RTSP_OUTPUT = _get_bool("SAVE_RTSP_OUTPUT", False)
LOG_DIR = _get("LOG_DIR", "logs")
CLIPS_DIR = _get("CLIPS_DIR", "output/staff_clips")
STAFF_CLIP_POST_FRAMES = max(1, _get_int("STAFF_CLIP_POST_FRAMES", 45))
STAFF_CLIP_FPS = max(1.0, _get_float("STAFF_CLIP_FPS", 15.0))

# Production pipeline (quality → track → collect → top-K → vote)
MIN_FACE_WIDTH = max(20.0, _get_float("MIN_FACE_WIDTH", 80.0))
MIN_BLUR_SCORE = max(1.0, _get_float("MIN_BLUR_SCORE", 60.0))
MAX_YAW_DEG = max(1.0, _get_float("MAX_YAW_DEG", 40.0))
MAX_PITCH_DEG = max(1.0, _get_float("MAX_PITCH_DEG", 30.0))
MAX_ROLL_DEG = max(1.0, _get_float("MAX_ROLL_DEG", 30.0))
COLLECT_FRAMES = max(1, _get_int("COLLECT_FRAMES", 8))
MIN_COLLECT_FRAMES = max(1, _get_int("MIN_COLLECT_FRAMES", 3))
TOP_QUALITY_FRAMES = max(1, _get_int("TOP_QUALITY_FRAMES", 3))
CONFIRM_VOTES = max(1, _get_int("CONFIRM_VOTES", 3))
TRACK_IOU = _get_float("TRACK_IOU", 0.25)
TRACK_MAX_MISSES = max(5, _get_int("TRACK_MAX_MISSES", 45))
MAX_EMBEDS_PER_PERSON = max(5, _get_int("MAX_EMBEDS_PER_PERSON", 50))

# RTSP decode / reconnect tuning
RTSP_BUFFER_SIZE = max(1, _get_int("RTSP_BUFFER_SIZE", 1))
RTSP_SKIP_BEFORE_RECONNECT = max(5, _get_int("RTSP_SKIP_BEFORE_RECONNECT", 45))
RTSP_RECONNECT_DELAY = max(0.5, _get_float("RTSP_RECONNECT_DELAY", 2.0))
RTSP_FFMPEG_OPTIONS = _get(
    "RTSP_FFMPEG_OPTIONS",
    # TCP + discard corrupt AU + low latency (handles missing picture warnings)
    "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay"
    "|max_delay;500000|stimeout;5000000|rw_timeout;5000000",
)
