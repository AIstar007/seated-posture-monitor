"""MediaPipe Tasks adapter implementing the PoseDetector port."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from seated_posture_monitor.application.ports import PoseObservation, RecoverableFrameError


class MediaPipePoseDetector:
    """CPU-only, single-person MediaPipe Pose Landmarker in VIDEO mode."""

    def __init__(
        self, model_path: Path,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        for name, value in (("detection_confidence", detection_confidence),
                            ("presence_confidence", presence_confidence),
                            ("tracking_confidence", tracking_confidence)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        self.detection_confidence = float(detection_confidence)
        self.presence_confidence = float(presence_confidence)
        self.tracking_confidence = float(tracking_confidence)
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_pose_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, bgr_frame: np.ndarray, timestamp_s: float) -> Optional[PoseObservation]:
        try:
            if bgr_frame.ndim != 3 or bgr_frame.shape[2] not in (3, 4):
                raise RecoverableFrameError("Expected a 3-channel BGR or 4-channel BGRA frame")
            conversion = cv2.COLOR_BGR2RGB if bgr_frame.shape[2] == 3 else cv2.COLOR_BGRA2RGB
            rgb = cv2.cvtColor(bgr_frame, conversion)
            rgb = np.ascontiguousarray(rgb)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(int(round(timestamp_s * 1000.0)), self._last_timestamp_ms + 1)
            result: Any = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except RecoverableFrameError:
            raise
        except (cv2.error, ValueError) as exc:
            raise RecoverableFrameError(f"MediaPipe rejected frame: {exc}") from exc
        self._last_timestamp_ms = timestamp_ms
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        return PoseObservation(result.pose_landmarks[0], result.pose_world_landmarks[0])

    def close(self) -> None:
        self._landmarker.close()
