"""Robust personal baseline calibration from stable, high-quality observations."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .models import PoseFeatures


@dataclass(frozen=True)
class CalibrationConfig:
    duration_s: float = 5.0
    minimum_samples: int = 20
    maximum_pitch_mad_deg: float = 4.0
    maximum_lateral_mad_ratio: float = 0.04
    maximum_roll_mad_deg: float = 4.0
    pitch_zero_score_delta_deg: float = 18.0
    lateral_zero_score_delta_ratio: float = 0.15
    shoulder_roll_zero_score_delta_deg: float = 15.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.minimum_samples < 5:
            raise ValueError("calibration duration must be positive and minimum_samples >= 5")
        if min(self.maximum_pitch_mad_deg, self.maximum_lateral_mad_ratio, self.maximum_roll_mad_deg) <= 0:
            raise ValueError("calibration stability limits must be positive")
        if min(
            self.pitch_zero_score_delta_deg,
            self.lateral_zero_score_delta_ratio,
            self.shoulder_roll_zero_score_delta_deg,
        ) <= 0:
            raise ValueError("calibration scoring deviation limits must be positive")


@dataclass(frozen=True)
class CalibrationBaseline:
    trunk_pitch_deg: float
    lateral_lean_ratio: float
    shoulder_roll_deg: float
    samples: int
    duration_s: float
    pitch_mad_deg: float
    lateral_mad_ratio: float
    shoulder_roll_mad_deg: float


class BaselineCalibrator:
    """Collect robust medians and reject unstable calibration segments."""

    def __init__(self, config: CalibrationConfig = CalibrationConfig()) -> None:
        self.config = config
        self._samples: list[tuple[float, PoseFeatures]] = []

    @property
    def sample_count(self) -> int:
        """Number of accepted, feature-valid calibration samples."""
        return len(self._samples)

    @property
    def sample_span_s(self) -> float:
        """Elapsed time covered by accepted calibration samples."""
        if len(self._samples) < 2:
            return 0.0
        times = [sample[0] for sample in self._samples]
        return max(times) - min(times)

    def add(self, timestamp_s: float, features: PoseFeatures) -> None:
        if not math.isfinite(timestamp_s):
            return
        self._samples.append((timestamp_s, features))

    @staticmethod
    def _mad(values: list[float], median: float) -> float:
        return statistics.median(abs(value - median) for value in values)

    def finish(self) -> CalibrationBaseline:
        if len(self._samples) < self.config.minimum_samples:
            raise ValueError(
                f"Calibration needs at least {self.config.minimum_samples} valid samples; "
                f"received {len(self._samples)}"
            )
        times = [sample[0] for sample in self._samples]
        duration = max(times) - min(times)
        if duration < self.config.duration_s * 0.8:
            raise ValueError("Calibration observations did not span enough time")
        pitches = [sample.trunk_pitch_deg for _, sample in self._samples]
        laterals = [sample.lateral_lean_ratio for _, sample in self._samples]
        rolls = [sample.shoulder_roll_deg for _, sample in self._samples]
        medians = (statistics.median(pitches), statistics.median(laterals), statistics.median(rolls))
        mads = (self._mad(pitches, medians[0]), self._mad(laterals, medians[1]), self._mad(rolls, medians[2]))
        limits = (self.config.maximum_pitch_mad_deg, self.config.maximum_lateral_mad_ratio, self.config.maximum_roll_mad_deg)
        if any(mad > limit for mad, limit in zip(mads, limits)):
            raise ValueError("Calibration was unstable; sit comfortably and retry")
        return CalibrationBaseline(*medians, len(self._samples), duration, *mads)
