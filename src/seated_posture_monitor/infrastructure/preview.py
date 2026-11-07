"""Optional OpenCV preview adapter; never required for headless operation."""
from __future__ import annotations

from typing import Any, Optional

import cv2
import math

from seated_posture_monitor.domain.models import PoseFeatures


class OpenCVPreview:
    def __init__(self, window_name: str = "Seated Posture Monitor", visibility_min: float = 0.55) -> None:
        self._window_name = window_name
        self._visibility_min = visibility_min

    def show(
        self,
        frame: Any,
        state: Optional[str],
        quality_score: Optional[float],
        features: Optional[PoseFeatures],
        status_detail: Optional[str] = None,
        observation: Any = None,
        image_quality: Any = None,
    ) -> bool:
        view = frame.copy()
        if observation is not None:
            self._draw_required_landmarks(view, observation.image_landmarks)
        colour = (50, 190, 70) if state == "good" else (30, 80, 230) if state == "bad" else (0, 190, 220)
        label = "CALIBRATING" if state == "calibrating" else state.upper() if state else "TRANSITION / UNKNOWN"
        cv2.putText(view, f"Posture: {label}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        if features is not None and quality_score is not None:
            detail = (
                f"Quality {quality_score:.0f}/100 | pitch {features.trunk_pitch_deg:.1f} deg "
                f"| landmark confidence {features.observation_confidence:.2f}"
            )
        else:
            detail = status_detail or "Pose unavailable"
        cv2.putText(view, detail[:100], (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 2)
        if image_quality is not None and image_quality.warnings:
            quality_text = "Image: " + ", ".join(image_quality.warnings)
            cv2.putText(view, quality_text[:100], (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 190, 220), 1)
        cv2.putText(view, "Press Q to stop", (18, view.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
        cv2.imshow(self._window_name, view)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def _draw_required_landmarks(self, frame: Any, landmarks: Any) -> None:
        height, width = frame.shape[:2]
        points = {}
        for index, label in ((11, "LS"), (12, "RS"), (23, "LH"), (24, "RH")):
            if index >= len(landmarks):
                continue
            landmark = landmarks[index]
            nx, ny = float(landmark.x), float(landmark.y)
            if not math.isfinite(nx) or not math.isfinite(ny):
                continue
            x, y = int(nx * width), int(ny * height)
            if 0 <= x < width and 0 <= y < height:
                points[index] = (x, y)
                confidence = getattr(landmark, "visibility", None)
                try:
                    confidence_value = float(confidence) if confidence is not None else 0.0
                except (TypeError, ValueError):
                    confidence_value = 0.0
                colour = (0, 210, 0) if confidence_value >= self._visibility_min else (0, 80, 255)
                cv2.circle(frame, (x, y), 6, colour, -1)
                cv2.putText(
                    frame, f"{label} {confidence_value:.2f}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1
                )
        for first, second in ((11, 12), (23, 24), (11, 23), (12, 24)):
            if first in points and second in points:
                cv2.line(frame, points[first], points[second], (0, 220, 0), 2)

    def close(self) -> None:
        cv2.destroyAllWindows()
