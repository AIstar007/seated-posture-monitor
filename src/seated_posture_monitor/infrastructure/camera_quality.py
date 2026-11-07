"""Lightweight, non-blocking image-quality diagnostics for camera frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    width: int
    height: int
    brightness_mean: float
    contrast_std: float
    sharpness_variance: float
    warnings: tuple[str, ...]


class CameraQualityAnalyzer:
    """Measure basic image conditions; thresholds are guidance, not gates."""

    def analyze(self, frame: Any) -> FrameQuality:
        if frame is None or getattr(frame, "ndim", 0) != 3 or frame.size == 0:
            raise ValueError("Cannot assess an empty or invalid camera frame")
        height, width = frame.shape[:2]
        if frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Expected BGR or BGRA camera frame")
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        warnings: list[str] = []
        if width < 640 or height < 480:
            warnings.append("low_resolution")
        if brightness < 45.0:
            warnings.append("low_light")
        elif brightness > 220.0:
            warnings.append("overexposed")
        if contrast < 18.0:
            warnings.append("low_contrast")
        # Laplacian variance depends on scene texture; use only a conservative hint.
        if sharpness < 25.0:
            warnings.append("possible_blur_or_soft_focus")
        return FrameQuality(
            width=width, height=height,
            brightness_mean=round(brightness, 2),
            contrast_std=round(contrast, 2),
            sharpness_variance=round(sharpness, 2),
            warnings=tuple(warnings),
        )
