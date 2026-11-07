"""Core posture rules; no camera, ML-framework, or CLI dependencies."""
from .geometry import DEFAULT_THRESHOLDS, PoseFeatureAnalysis, analyze_pose_features, extract_pose_features
from .models import BreakEvent, GeometryThresholds, PoseFeatures, RunCounters, StabilizerConfig
from .session import SessionAccumulator
from .temporal import TemporalStabilizer

__all__ = [
    "BreakEvent", "DEFAULT_THRESHOLDS", "GeometryThresholds", "PoseFeatures", "PoseFeatureAnalysis",
    "RunCounters", "SessionAccumulator", "StabilizerConfig", "TemporalStabilizer",
    "analyze_pose_features", "extract_pose_features",
]
