"""Input configuration for one monitoring use case."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonitorConfig:
    source_label: str
    duration_s: float = 60.0
    output_path: Path = Path("posture.json")
    visibility_min: float = 0.55
    sample_fps: float = 10.0
    preview_enabled: bool = False
    model_name: str = "pose_landmarker_lite.task"
    calibration_enabled: bool = False
    calibration_duration_s: float = 5.0
    calibration_acquisition_timeout_s: float = 20.0
    camera_view: str = "unspecified"
    pose_detection_confidence: float = 0.5
    pose_presence_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        if not 0.0 <= self.visibility_min <= 1.0:
            raise ValueError("visibility_min must be between 0 and 1")
        if not 0.0 < self.sample_fps <= 60.0:
            raise ValueError("sample_fps must be greater than 0 and at most 60")
        if self.calibration_duration_s <= 0:
            raise ValueError("calibration_duration_s must be positive")
        if self.calibration_acquisition_timeout_s <= 0:
            raise ValueError("calibration_acquisition_timeout_s must be positive")
        if self.camera_view not in {"unspecified", "frontal", "45-degree", "side"}:
            raise ValueError("camera_view must be unspecified, frontal, 45-degree or side")
        for name in ("pose_detection_confidence", "pose_presence_confidence", "pose_tracking_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.calibration_enabled and self.sample_fps * self.calibration_duration_s < 5.0:
            raise ValueError("calibration settings must allow at least five samples")
