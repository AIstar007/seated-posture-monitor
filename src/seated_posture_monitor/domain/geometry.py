"""Pure pose geometry and posture-quality calculation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .models import GeometryThresholds, PoseFeatures

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
REQUIRED_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
DEFAULT_THRESHOLDS = GeometryThresholds()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def linear_goodness(value: float, good_at: float, bad_at: float) -> float:
    """Return 100 at/below ``good_at`` and 0 at/above ``bad_at``."""
    if bad_at <= good_at:
        raise ValueError("bad_at must be greater than good_at")
    return 100.0 * clamp((bad_at - value) / (bad_at - good_at))


def _xyz(landmark: Any) -> tuple[float, float, float]:
    return float(landmark.x), float(landmark.y), float(landmark.z)


@dataclass(frozen=True)
class PoseFeatureAnalysis:
    """Feature extraction result with an auditable rejection reason."""

    features: Optional[PoseFeatures]
    reason: Optional[str] = None
    minimum_landmark_confidence: Optional[float] = None


def analyze_pose_features(
    image_landmarks: Sequence[Any],
    world_landmarks: Sequence[Any],
    thresholds: GeometryThresholds = DEFAULT_THRESHOLDS,
) -> PoseFeatureAnalysis:
    """Extract posture features and explain why an observation was rejected.

    The domain remains framework-independent: MediaPipe-like objects are read
    through their documented attributes without importing MediaPipe.
    """
    max_index = max(REQUIRED_LANDMARKS)
    if len(image_landmarks) <= max_index or len(world_landmarks) <= max_index:
        return PoseFeatureAnalysis(None, "landmark_list_incomplete")

    def confidence(landmark: Any) -> tuple[Optional[float], Optional[str]]:
        values = []
        for attribute in ("visibility", "presence"):
            raw_value = getattr(landmark, attribute, None)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return None, f"invalid_{attribute}"
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return None, f"invalid_{attribute}"
            values.append(value)
        if not values:
            return None, "confidence_metadata_missing"
        return min(values), None

    names = {
        LEFT_SHOULDER: "left_shoulder",
        RIGHT_SHOULDER: "right_shoulder",
        LEFT_HIP: "left_hip",
        RIGHT_HIP: "right_hip",
    }
    confidences: list[float] = []
    rejected: list[str] = []
    for index in REQUIRED_LANDMARKS:
        image_conf, image_error = confidence(image_landmarks[index])
        world_conf, world_error = confidence(world_landmarks[index])
        if image_error:
            rejected.append(f"{names[index]}_image_{image_error}")
        if world_error:
            rejected.append(f"{names[index]}_world_{world_error}")
        if image_error or world_error:
            continue
        assert image_conf is not None and world_conf is not None
        value = min(image_conf, world_conf)
        confidences.append(value)
        if image_conf < thresholds.visibility_min:
            rejected.append(f"{names[index]}_image_visibility_low_{image_conf:.2f}")
        if world_conf < thresholds.visibility_min:
            rejected.append(f"{names[index]}_world_visibility_low_{world_conf:.2f}")
    min_confidence = min(confidences) if confidences else None
    if rejected:
        return PoseFeatureAnalysis(
            None, "low_or_missing_landmark_confidence:" + ",".join(rejected), min_confidence
        )

    try:
        left_shoulder = _xyz(world_landmarks[LEFT_SHOULDER])
        right_shoulder = _xyz(world_landmarks[RIGHT_SHOULDER])
        left_hip = _xyz(world_landmarks[LEFT_HIP])
        right_hip = _xyz(world_landmarks[RIGHT_HIP])
    except (AttributeError, IndexError, TypeError, ValueError):
        return PoseFeatureAnalysis(None, "landmark_coordinates_missing", min_confidence)

    coordinates = (*left_shoulder, *right_shoulder, *left_hip, *right_hip)
    if not all(math.isfinite(value) for value in coordinates):
        return PoseFeatureAnalysis(None, "non_finite_landmark_coordinates", min_confidence)

    shoulder_mid = tuple((a + b) / 2.0 for a, b in zip(left_shoulder, right_shoulder))
    hip_mid = tuple((a + b) / 2.0 for a, b in zip(left_hip, right_hip))
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    dz = shoulder_mid[2] - hip_mid[2]
    torso_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(torso_length) or torso_length < 1e-4:
        return PoseFeatureAnalysis(None, "degenerate_torso_geometry", min_confidence)

    trunk_pitch_deg = math.degrees(math.atan2(abs(dz), max(abs(dy), 1e-6)))
    lateral_lean_ratio = abs(dx) / torso_length
    shoulder_dx = right_shoulder[0] - left_shoulder[0]
    shoulder_dy = right_shoulder[1] - left_shoulder[1]
    shoulder_span = math.hypot(shoulder_dx, shoulder_dy)
    if shoulder_span < 1e-4:
        return PoseFeatureAnalysis(None, "degenerate_shoulder_span", min_confidence)
    shoulder_roll_deg = math.degrees(math.atan2(abs(shoulder_dy), abs(shoulder_dx)))

    pitch_score = linear_goodness(trunk_pitch_deg, thresholds.pitch_good_deg, thresholds.pitch_bad_deg)
    lateral_score = linear_goodness(lateral_lean_ratio, thresholds.lateral_good_ratio, thresholds.lateral_bad_ratio)
    shoulder_score = linear_goodness(shoulder_roll_deg, thresholds.shoulder_roll_good_deg, thresholds.shoulder_roll_bad_deg)
    quality = (
        thresholds.pitch_weight * pitch_score
        + thresholds.lateral_weight * lateral_score
        + thresholds.shoulder_weight * shoulder_score
    )
    values = (trunk_pitch_deg, lateral_lean_ratio, shoulder_roll_deg, quality)
    if not all(math.isfinite(value) for value in values):
        return PoseFeatureAnalysis(None, "non_finite_posture_geometry", min_confidence)

    features = PoseFeatures(
        trunk_pitch_deg=round(trunk_pitch_deg, 2),
        lateral_lean_ratio=round(lateral_lean_ratio, 4),
        shoulder_roll_deg=round(shoulder_roll_deg, 2),
        quality_score=round(clamp(quality, 0.0, 100.0), 2),
        observation_confidence=round(sum(confidences) / len(confidences), 4),
    )
    return PoseFeatureAnalysis(features, None, min_confidence)


def extract_pose_features(
    image_landmarks: Sequence[Any],
    world_landmarks: Sequence[Any],
    thresholds: GeometryThresholds = DEFAULT_THRESHOLDS,
) -> Optional[PoseFeatures]:
    """Backward-compatible convenience API returning features or ``None``."""
    return analyze_pose_features(image_landmarks, world_landmarks, thresholds).features

