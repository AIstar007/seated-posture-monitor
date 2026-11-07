"""Time-aware EMA, hysteresis and candidate transition timing."""
from __future__ import annotations
import math
from typing import Optional
from .models import StabilizerConfig


class TemporalStabilizer:
    """Reject transient noise and expose the start time of confirmed transitions."""
    def __init__(self, config: StabilizerConfig = StabilizerConfig()) -> None:
        self.config = config
        self.ema_score: Optional[float] = None
        self.state: Optional[str] = None
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self.last_transition: Optional[tuple[str, float]] = None

    def reset(self) -> None:
        self.ema_score = None; self.state = None; self._candidate = None
        self._candidate_since = None; self._last_timestamp = None
        self.last_transition = None

    def note_missing(self, timestamp_s: float) -> Optional[str]:
        if self._last_timestamp is not None and timestamp_s - self._last_timestamp > self.config.max_observation_gap_s:
            self.reset()
        return self.state

    def update(self, timestamp_s: float, quality_score: float) -> Optional[str]:
        self.last_transition = None
        if not math.isfinite(timestamp_s) or not math.isfinite(quality_score) or not 0 <= quality_score <= 100:
            return self.state
        if self._last_timestamp is None:
            self.ema_score = quality_score
        else:
            dt = timestamp_s - self._last_timestamp
            if dt <= 0: return self.state
            if dt > self.config.max_observation_gap_s:
                self.state = None; self.ema_score = quality_score
                self._candidate = None; self._candidate_since = None
                self._last_timestamp = timestamp_s
                return self.state
            alpha = 1.0 - math.exp(-dt / self.config.ema_time_constant_s)
            self.ema_score = (1-alpha) * float(self.ema_score) + alpha * quality_score
        self._last_timestamp = timestamp_s
        if self.ema_score <= self.config.bad_enter_score: desired = "bad"
        elif self.ema_score >= self.config.good_enter_score: desired = "good"
        else: desired = None
        if desired is None or desired == self.state:
            self._candidate = None; self._candidate_since = None
        elif desired != self._candidate:
            self._candidate = desired; self._candidate_since = timestamp_s
        elif self._candidate_since is not None and timestamp_s - self._candidate_since >= self.config.state_confirmation_s:
            transition_at = self._candidate_since
            self.state = desired
            self.last_transition = (desired, transition_at)
            self._candidate = None; self._candidate_since = None
        return self.state
