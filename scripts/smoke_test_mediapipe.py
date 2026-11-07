"""Download the pinned model and execute one real CPU inference for CI."""
from __future__ import annotations

import sys
from pathlib import Path

# The project uses a src/ layout. When this file is executed directly as
# `python scripts/smoke_test_mediapipe.py`, Python puts `scripts/` rather than
# the repository root on sys.path. Add the local source directory so the
# smoke test works from a fresh checkout without requiring `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from seated_posture_monitor.infrastructure.mediapipe_detector import MediaPipePoseDetector
from seated_posture_monitor.infrastructure.model_asset import ensure_model


def main() -> None:
    model_path = ensure_model()
    detector = MediaPipePoseDetector(model_path)
    try:
        # A blank frame should normally produce no pose; the goal is to prove
        # that model loading and the CPU VIDEO inference path execute cleanly.
        detector.detect(np.zeros((256, 256, 3), dtype=np.uint8), timestamp_s=0.033)
    finally:
        detector.close()
    print("MediaPipe model loaded and CPU inference completed.")


if __name__ == "__main__":
    main()
