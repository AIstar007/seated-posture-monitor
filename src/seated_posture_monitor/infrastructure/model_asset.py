"""Manage the official MediaPipe Lite model asset."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from seated_posture_monitor import __version__ as APP_VERSION

MODEL_VERSION = "1"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_lite/float16/{MODEL_VERSION}/pose_landmarker_lite.task"
)
# Include the model asset version in the cache name so an older `latest`
# download cannot be mistaken for the pinned asset.
MODEL_FILENAME = f"pose_landmarker_lite-v{MODEL_VERSION}.task"


def default_model_path() -> Path:
    return Path.home() / ".cache" / "seated-posture-monitor" / MODEL_FILENAME


def ensure_model(model_path: Path | None = None) -> Path:
    """Resolve a supplied model or download the official Lite model atomically."""
    path = model_path.expanduser() if model_path else default_model_path()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    if model_path is not None:
        raise FileNotFoundError(f"Model file not found or invalid: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": f"seated-posture-monitor/{APP_VERSION}"}
        )
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if partial.stat().st_size <= 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small; refusing to use it")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return path
