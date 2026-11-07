"""Small interfaces that isolate the use case from external technologies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence


class RecoverableFrameError(Exception):
    """An isolated frame could not be decoded or passed to inference."""


class FrameSource(Protocol):
    """Video input port. Implementations may wrap webcams, files or test data."""

    @property
    def is_camera(self) -> bool: ...

    def is_opened(self) -> bool: ...
    def read(self) -> tuple[bool, Any]: ...
    def fps(self) -> float: ...
    def frame_count(self) -> float: ...
    def position_seconds(self) -> float: ...
    def position_frames(self) -> float: ...
    def seek_frame(self, frame_number: int) -> bool: ...
    def release(self) -> None: ...


class PoseDetector(Protocol):
    """Pose-estimation port; concrete ML frameworks stay outside the use case."""

    def detect(self, frame: Any, timestamp_s: float) -> Optional["PoseObservation"]: ...
    def close(self) -> None: ...


class Preview(Protocol):
    """Optional presentation port. Returning True requests a graceful stop."""

    def show(
        self,
        frame: Any,
        state: Optional[str],
        quality_score: Optional[float],
        features: Any,
        status_detail: Optional[str] = None,
        observation: Optional["PoseObservation"] = None,
        image_quality: Any = None,
    ) -> bool: ...

    def close(self) -> None: ...


class ReportWriter(Protocol):
    """Output port for a completed JSON-compatible summary."""

    def write(self, destination: Path, summary: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PoseObservation:
    """Framework-neutral view of the landmarks needed by the domain."""

    image_landmarks: Sequence[Any]
    world_landmarks: Sequence[Any]
