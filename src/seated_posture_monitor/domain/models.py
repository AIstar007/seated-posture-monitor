"""Domain models and policy configuration for posture analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PoseFeatures:
    """Scale-independent measurements extracted from one pose observation."""

    trunk_pitch_deg: float
    lateral_lean_ratio: float
    shoulder_roll_deg: float
    quality_score: float
    observation_confidence: float = 1.0


@dataclass
class BreakEvent:
    """A confirmed interval during which posture was classified as poor."""

    start_s: float
    end_s: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass
class RunCounters:
    frames_read: int = 0
    frames_sampled: int = 0
    # pose_detected_frames counts samples with valid posture geometry (kept for
    # backward compatibility); model detections and geometry acceptance are
    # tracked separately so poor framing is not confused with no-person results.
    pose_detected_frames: int = 0
    pose_model_detected_frames: int = 0
    frames_without_pose: int = 0
    frames_pose_not_detected: int = 0
    frames_geometry_rejected: int = 0
    geometry_rejection_reasons: dict[str, int] = field(default_factory=dict)
    geometry_rejection_examples: list[str] = field(default_factory=list)
    dropped_or_corrupt_frames: int = 0
    recoverable_inference_errors: int = 0


@dataclass(frozen=True)
class GeometryThresholds:
    """Interpretable starting thresholds; calibrate against labelled sessions."""

    pitch_good_deg: float = 10.0
    pitch_bad_deg: float = 25.0
    lateral_good_ratio: float = 0.05
    lateral_bad_ratio: float = 0.18
    shoulder_roll_good_deg: float = 5.0
    shoulder_roll_bad_deg: float = 18.0
    pitch_weight: float = 0.65
    lateral_weight: float = 0.20
    shoulder_weight: float = 0.15
    visibility_min: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.visibility_min <= 1.0:
            raise ValueError("visibility_min must be between 0 and 1")
        if self.pitch_bad_deg <= self.pitch_good_deg:
            raise ValueError("pitch_bad_deg must exceed pitch_good_deg")
        if self.lateral_bad_ratio <= self.lateral_good_ratio:
            raise ValueError("lateral_bad_ratio must exceed lateral_good_ratio")
        if self.shoulder_roll_bad_deg <= self.shoulder_roll_good_deg:
            raise ValueError("shoulder_roll_bad_deg must exceed shoulder_roll_good_deg")
        weights = (self.pitch_weight, self.lateral_weight, self.shoulder_weight)
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("feature weights must be non-negative and sum to 1")


@dataclass(frozen=True)
class StabilizerConfig:
    ema_time_constant_s: float = 0.65
    bad_enter_score: float = 58.0
    good_enter_score: float = 72.0
    state_confirmation_s: float = 0.75
    max_observation_gap_s: float = 1.0

    def __post_init__(self) -> None:
        if self.ema_time_constant_s <= 0:
            raise ValueError("ema_time_constant_s must be positive")
        if not 0 <= self.bad_enter_score < self.good_enter_score <= 100:
            raise ValueError("expected 0 <= bad threshold < good threshold <= 100")
        if self.state_confirmation_s < 0 or self.max_observation_gap_s <= 0:
            raise ValueError("confirmation must be non-negative and max gap positive")
