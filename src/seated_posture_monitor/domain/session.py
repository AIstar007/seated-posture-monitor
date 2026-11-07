"""Session-level aggregation with retroactive, time-accurate state accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .geometry import clamp
from .models import PoseFeatures


@dataclass(frozen=True)
class _Interval:
    start_s: float
    end_s: float
    state: Optional[str]
    quality: Optional[float]
    confidence: Optional[float]


class SessionAccumulator:
    """Integrate stable states and revise candidate intervals on confirmation.

    The temporal filter confirms a transition only after it has persisted. Its
    ``transition_at`` is therefore earlier than the confirming observation.
    We retain a small interval timeline and reclassify that candidate period so
    duration metrics and break events agree with one another.
    """

    def __init__(self, start_s: float = 0.0, max_gap_s: float = 1.0) -> None:
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        self.start_s = float(start_s)
        self.max_gap_s = float(max_gap_s)
        self.last_timestamp = float(start_s)
        self.last_state: Optional[str] = None
        self.last_quality: Optional[float] = None
        self.last_confidence: Optional[float] = None
        self._intervals: list[_Interval] = []
        self._feature_stats: dict[str, dict[str, float]] = {}

    def record_features(self, features: PoseFeatures) -> None:
        """Record raw feature measurements for auditable session statistics."""
        values = {
            "trunk_pitch_deg": features.trunk_pitch_deg,
            "lateral_lean_ratio": features.lateral_lean_ratio,
            "shoulder_roll_deg": features.shoulder_roll_deg,
        }
        for name, value in values.items():
            stats = self._feature_stats.setdefault(
                name, {"count": 0.0, "sum": 0.0, "min": value, "max": value}
            )
            stats["count"] += 1.0
            stats["sum"] += value
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)

    def _append(
        self,
        start_s: float,
        end_s: float,
        state: Optional[str],
        quality: Optional[float],
        confidence: Optional[float],
    ) -> None:
        if end_s - start_s > 1e-9:
            self._intervals.append(
                _Interval(start_s, end_s, state, quality, confidence)
            )

    def _reclassify(self, start_s: float, end_s: float, state: str) -> None:
        """Change interval labels over [start_s, end_s], preserving measurements."""
        if state not in {"good", "bad"} or end_s <= start_s:
            return
        revised: list[_Interval] = []
        for interval in self._intervals:
            overlap_start = max(interval.start_s, start_s)
            overlap_end = min(interval.end_s, end_s)
            if overlap_end <= overlap_start:
                revised.append(interval)
                continue
            if interval.start_s < overlap_start:
                revised.append(
                    _Interval(
                        interval.start_s,
                        overlap_start,
                        interval.state,
                        interval.quality,
                        interval.confidence,
                    )
                )
            revised.append(
                _Interval(
                    overlap_start,
                    overlap_end,
                    state,
                    interval.quality,
                    interval.confidence,
                )
            )
            if overlap_end < interval.end_s:
                revised.append(
                    _Interval(
                        overlap_end,
                        interval.end_s,
                        interval.state,
                        interval.quality,
                        interval.confidence,
                    )
                )
        self._intervals = revised

    def update(
        self,
        timestamp_s: float,
        state: Optional[str],
        quality_score: Optional[float],
        transition_at: Optional[float] = None,
        observation_confidence: Optional[float] = None,
    ) -> None:
        if timestamp_s <= self.last_timestamp:
            return
        if state not in {None, "good", "bad"}:
            raise ValueError("state must be 'good', 'bad' or None")

        previous_state = self.last_state
        dt = timestamp_s - self.last_timestamp
        if dt > self.max_gap_s:
            trusted_end = self.last_timestamp + self.max_gap_s
            self._append(
                self.last_timestamp,
                trusted_end,
                self.last_state,
                self.last_quality,
                self.last_confidence,
            )
            self._append(trusted_end, timestamp_s, None, None, None)
        else:
            self._append(
                self.last_timestamp,
                timestamp_s,
                self.last_state,
                self.last_quality,
                self.last_confidence,
            )
            # The candidate start can be several observations in the past.
            # A large gap is deliberately not retroactively classified.
            if (
                state in {"good", "bad"}
                and state != previous_state
                and transition_at is not None
            ):
                candidate_start = max(self.start_s, min(timestamp_s, transition_at))
                self._reclassify(candidate_start, timestamp_s, state)

        self.last_timestamp = timestamp_s
        self.last_state = state
        self.last_quality = quality_score
        self.last_confidence = observation_confidence

    def finish(self, end_s: float) -> None:
        if end_s <= self.last_timestamp:
            return
        dt = end_s - self.last_timestamp
        trusted_end = min(end_s, self.last_timestamp + self.max_gap_s)
        self._append(
            self.last_timestamp,
            trusted_end,
            self.last_state,
            self.last_quality,
            self.last_confidence,
        )
        if trusted_end < end_s:
            self._append(trusted_end, end_s, None, None, None)
        self.last_timestamp = end_s

    def summary(self, session_duration_s: float) -> dict[str, Any]:
        duration = max(float(session_duration_s), 0.0)
        good_s = sum(i.end_s - i.start_s for i in self._intervals if i.state == "good")
        bad_s = sum(i.end_s - i.start_s for i in self._intervals if i.state == "bad")
        classified_s = good_s + bad_s
        unclassified_s = max(0.0, duration - classified_s)

        quality_parts = [i for i in self._intervals if i.quality is not None]
        quality_duration = sum(i.end_s - i.start_s for i in quality_parts)
        quality_integral = sum(
            float(i.quality) * (i.end_s - i.start_s) for i in quality_parts
        )
        confidence_parts = [i for i in self._intervals if i.confidence is not None]
        confidence_duration = sum(i.end_s - i.start_s for i in confidence_parts)
        confidence_integral = sum(
            float(i.confidence) * (i.end_s - i.start_s) for i in confidence_parts
        )
        mean_quality = quality_integral / quality_duration if quality_duration else None
        mean_confidence = (
            confidence_integral / confidence_duration if confidence_duration else None
        )

        breaks: list[dict[str, float]] = []
        active: Optional[dict[str, float]] = None
        for interval in self._intervals:
            if interval.state == "bad":
                if active is not None and interval.start_s <= active["end_s"] + 1e-6:
                    active["end_s"] = interval.end_s
                else:
                    if active is not None:
                        active["duration_s"] = active["end_s"] - active["start_s"]
                        breaks.append(active)
                    active = {"start_s": interval.start_s, "end_s": interval.end_s}
            elif active is not None:
                active["duration_s"] = active["end_s"] - active["start_s"]
                breaks.append(active)
                active = None
        if active is not None:
            active["duration_s"] = active["end_s"] - active["start_s"]
            breaks.append(active)

        for event in breaks:
            for key in ("start_s", "end_s", "duration_s"):
                event[key] = round(event[key], 2)

        good_pct = 100.0 * good_s / duration if duration > 0 else 0.0
        classified_good_pct = 100.0 * good_s / classified_s if classified_s > 0 else 0.0
        coverage_pct = 100.0 * classified_s / duration if duration > 0 else 0.0
        feature_statistics = {
            name: {
                "samples": int(values["count"]),
                "mean": round(values["sum"] / values["count"], 3),
                "min": round(values["min"], 3),
                "max": round(values["max"], 3),
            }
            for name, values in self._feature_stats.items()
            if values["count"] > 0
        }

        return {
            "session_duration_s": round(duration, 2),
            "feature_statistics": feature_statistics,
            "good_posture_duration_s": round(good_s, 2),
            "bad_posture_duration_s": round(bad_s, 2),
            "unclassified_duration_s": round(unclassified_s, 2),
            "good_posture_pct": round(clamp(good_pct, 0.0, 100.0), 2),
            "good_posture_pct_of_classified_time": round(
                clamp(classified_good_pct, 0.0, 100.0), 2
            ),
            "classified_time_coverage_pct": round(
                clamp(coverage_pct, 0.0, 100.0), 2
            ),
            # Kept as a compatibility alias for the V1.2 JSON field.
            "stable_observation_coverage_pct": round(
                clamp(coverage_pct, 0.0, 100.0), 2
            ),
            "mean_frame_quality": round(mean_quality, 2) if mean_quality is not None else None,
            "mean_landmark_confidence": (
                round(mean_confidence, 4) if mean_confidence is not None else None
            ),
            "posture_score_0_100": round(clamp(good_pct, 0.0, 100.0), 1),
            "posture_break_count": len(breaks),
            "posture_breaks": breaks,
        }
