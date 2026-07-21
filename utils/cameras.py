"""Load RTSP / MP4 sources from cameras.yaml (separate sections)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import yaml

from utils.config import ROOT, _get, INPUT_DIR


@dataclass
class Camera:
    id: str
    name: str
    enabled: bool
    kind: str  # "rtsp" | "mp4"
    url: str = ""
    path: str = ""

    @property
    def source(self) -> str:
        return self.url if self.kind == "rtsp" else self.path

    def masked_url(self) -> str:
        value = self.source
        if self.kind != "rtsp" or "@" not in value:
            return value
        prefix, rest = value.split("@", 1)
        if "://" not in prefix:
            return value
        scheme, _creds = prefix.split("://", 1)
        return f"{scheme}://***:***@{rest}"


def _default_cameras_path() -> Path:
    configured = _get("CAMERAS_FILE", "cameras.yaml")
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _build_url_from_channel(channel: int, subtype: int) -> str:
    username = _get("RTSP_USERNAME")
    password = _get("RTSP_PASSWORD")
    host = _get("RTSP_HOST")
    port = _get("RTSP_PORT", "554")

    if not host:
        raise ValueError(
            "Camera uses channel= but RTSP_HOST is missing in .env"
        )

    userinfo = ""
    if username:
        userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}@"

    path = f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    return f"rtsp://{userinfo}{host}:{port}{path}"


def _normalize_rtsp_url(url: str) -> str:
    """
    Fix common password typo: user:pass@@host  (password ends with @)
    → user:pass%40@host
    """
    url = url.strip()
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@@" in rest and rest.count("@") >= 2:
        # admin:RajaGul786@@97.176... → encode trailing @ in password
        creds, hostpart = rest.split("@@", 1)
        if ":" in creds:
            user, password = creds.split(":", 1)
            return (
                f"{scheme}://{quote(user, safe='')}:"
                f"{quote(password + '@', safe='')}@{hostpart}"
            )
    return url


def _resolve_rtsp(entry: dict) -> str:
    url = (entry.get("url") or "").strip()
    if url:
        return _normalize_rtsp_url(url)

    if "channel" not in entry or entry.get("channel") is None:
        raise ValueError(
            f"RTSP camera '{entry.get('id')}' needs either 'url' or 'channel'"
        )

    channel = int(entry["channel"])
    subtype = int(entry.get("subtype", 1))
    return _build_url_from_channel(channel, subtype)


def _parse_entries(raw_list, kind: str) -> list[Camera]:
    cameras: list[Camera] = []
    seen_ids: set[str] = set()

    if not isinstance(raw_list, list):
        return cameras

    for entry in raw_list:
        if not isinstance(entry, dict):
            continue

        cam_id = str(entry.get("id") or "").strip()
        if not cam_id:
            raise ValueError(f"Each {kind} entry needs a non-empty 'id'")
        if cam_id in seen_ids:
            raise ValueError(f"Duplicate {kind} id: {cam_id}")
        seen_ids.add(cam_id)

        if kind == "rtsp":
            cameras.append(
                Camera(
                    id=cam_id,
                    name=str(entry.get("name") or cam_id).strip(),
                    enabled=bool(entry.get("enabled", True)),
                    kind="rtsp",
                    url=_resolve_rtsp(entry),
                )
            )
        else:
            path = str(entry.get("path") or "").strip()
            if not path:
                raise ValueError(f"MP4 entry '{cam_id}' needs a 'path'")
            cameras.append(
                Camera(
                    id=cam_id,
                    name=str(entry.get("name") or cam_id).strip(),
                    enabled=bool(entry.get("enabled", True)),
                    kind="mp4",
                    path=path,
                )
            )

    return cameras


def load_cameras(
    source: str = "rtsp",
    path: Path | None = None,
) -> list[Camera]:
    """
    Load the `rtsp:` or `mp4:` section from cameras.yaml.
    Also accepts legacy top-level `cameras:` as RTSP for old files.
    """
    source = source.lower().strip()
    if source not in {"rtsp", "mp4"}:
        raise ValueError("source must be 'rtsp' or 'mp4'")

    cameras_path = path or _default_cameras_path()
    if not cameras_path.is_file():
        raise FileNotFoundError(
            f"Cameras file not found: {cameras_path}\n"
            "Create cameras.yaml or set CAMERAS_FILE in .env"
        )

    with open(cameras_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if source in data:
        raw_list = data.get(source) or []
    elif source == "rtsp" and data.get("cameras"):
        # Legacy format
        raw_list = data.get("cameras") or []
    else:
        raw_list = []

    cameras = _parse_entries(raw_list, source)
    if not cameras:
        raise ValueError(
            f"No '{source}' entries in {cameras_path}. "
            f"Add items under the `{source}:` section."
        )
    return cameras


def select_cameras(
    cameras: list[Camera],
    selected_ids: Iterable[str] | None = None,
    source: str = "rtsp",
) -> list[Camera]:
    """
    If selected_ids is empty/None -> all enabled cameras.
    If selected_ids is set -> those ids only (even if enabled: false).
    """
    ids = [item.strip() for item in (selected_ids or []) if item.strip()]
    if not ids:
        chosen = [cam for cam in cameras if cam.enabled]
        if not chosen:
            lines = [
                f"No enabled {source} entries in cameras.yaml.",
                "Fix one of these:",
                f"  1) Set enabled: true under `{source}:` and set a real path/url",
                f"  2) Force one entry: --cameras <id>",
                "",
                f"Available {source} entries:",
            ]
            for cam in cameras:
                flag = "enabled" if cam.enabled else "disabled"
                loc = cam.path if cam.kind == "mp4" else cam.masked_url()
                lines.append(f"  {cam.id:<8} ({flag:<8}) {cam.name} -> {loc}")
            raise ValueError("\n".join(lines))
        return chosen

    by_id = {cam.id: cam for cam in cameras}
    missing = [cam_id for cam_id in ids if cam_id not in by_id]
    if missing:
        known = ", ".join(cam.id for cam in cameras)
        raise ValueError(
            f"Unknown id(s): {', '.join(missing)}. Known: {known}"
        )
    return [by_id[cam_id] for cam_id in ids]


def discover_mp4_in_folder(folder: str | None = None) -> list[Camera]:
    """
    Fallback when no mp4: entries are enabled in yaml.
    Picks up every .mp4 in input/ (or INPUT_DIR from .env).
    """
    base = Path(folder or INPUT_DIR)
    if not base.is_absolute():
        base = ROOT / base
    if not base.is_dir():
        return []

    cameras: list[Camera] = []
    for idx, path in enumerate(sorted(base.glob("*.mp4")), start=1):
        cameras.append(
            Camera(
                id=f"input{idx}",
                name=path.stem,
                enabled=True,
                kind="mp4",
                path=str(path),
            )
        )
    return cameras
