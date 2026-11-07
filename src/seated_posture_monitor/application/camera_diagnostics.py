"""Diagnostic use case to separate image, detector and geometry failures."""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Callable

from seated_posture_monitor.domain.geometry import analyze_pose_features
from seated_posture_monitor.domain.models import GeometryThresholds
from .ports import RecoverableFrameError


class CameraDiagnosticUseCase:
    def __init__(
        self, frame_source: Any, pose_detector: Any,
        quality_analyzer: Any,
        preview: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = frame_source
        self._detector = pose_detector
        self._quality = quality_analyzer
        self._preview = preview
        self._monotonic = monotonic

    def run(
        self, duration_s: float = 8.0, sample_fps: float = 5.0,
        visibility_min: float = 0.55,
        thresholds: GeometryThresholds | None = None,
    ) -> dict[str, Any]:
        if duration_s <= 0 or sample_fps <= 0:
            raise ValueError("duration_s and sample_fps must be positive")
        thresholds = thresholds or GeometryThresholds(visibility_min=visibility_min)
        started = self._monotonic()
        last_sample = -math.inf
        frame_count = sampled_count = pose_count = accepted_count = 0
        qualities: list[Any] = []
        reject_reasons: Counter[str] = Counter()
        no_pose = 0
        image_quality_errors = 0
        last_frame = None
        # Give a 30-FPS camera enough reads to run for the requested duration
        # even when inference sampling is intentionally much slower.
        try:
            source_fps = float(self._source.fps())
        except (AttributeError, TypeError, ValueError):
            source_fps = 0.0
        if not math.isfinite(source_fps) or source_fps <= 1.0 or source_fps > 240.0:
            source_fps = 30.0
        max_frames = max(30, int(duration_s * max(source_fps * 1.5, sample_fps * 4.0)))
        consecutive_read_failures = 0
        try:
            while self._monotonic() - started < duration_s and frame_count < max_frames:
                ok, frame = self._source.read()
                if not ok:
                    if self._source.is_camera:
                        consecutive_read_failures += 1
                        time.sleep(0.05)
                        if consecutive_read_failures >= 10:
                            break
                        continue
                    break
                consecutive_read_failures = 0
                frame_count += 1
                now = self._monotonic() - started
                if now - last_sample < 1.0 / sample_fps:
                    continue
                last_sample = now
                sampled_count += 1
                try:
                    qualities.append(self._quality.analyze(frame))
                except (TypeError, ValueError, AttributeError):
                    image_quality_errors += 1
                try:
                    observation = self._detector.detect(frame, now)
                except RecoverableFrameError as exc:
                    reject_reasons[f"recoverable_inference_error:{type(exc).__name__}"] += 1
                    continue
                if observation is None:
                    no_pose += 1
                    if self._preview is not None and qualities:
                        if self._preview.show(frame, None, None, None, "NO PERSON POSE DETECTED", None, qualities[-1]):
                            break
                    continue
                pose_count += 1
                analysis = analyze_pose_features(
                    observation.image_landmarks, observation.world_landmarks, thresholds
                )
                if analysis.features is None:
                    reject_reasons[analysis.reason or "geometry_rejected"] += 1
                    preview_state = None
                    preview_detail = f"LANDMARKS REJECTED: {analysis.reason or 'unknown'}"
                    preview_features = None
                else:
                    accepted_count += 1
                    preview_state = "good" if analysis.features.quality_score >= 72 else "bad" if analysis.features.quality_score < 58 else None
                    preview_detail = "POSE GEOMETRY VALID"
                    preview_features = analysis.features
                if self._preview is not None and qualities:
                    if self._preview.show(
                        frame, preview_state,
                        preview_features.quality_score if preview_features else None,
                        preview_features, preview_detail, observation, qualities[-1]
                    ):
                        break
        finally:
            for resource, method in ((self._preview, "close"), (self._detector, "close"), (self._source, "release")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception:
                        pass

        if not qualities:
            return {
                "status": "failed", "reason": "no_image_quality_samples",
                "frames_read": frame_count, "frames_sampled": sampled_count,
                "image_quality_errors": image_quality_errors,
                "pose_detections": pose_count,
                "usable_geometry_samples": accepted_count,
            }
        mean = lambda attr: round(sum(getattr(item, attr) for item in qualities) / len(qualities), 2)
        warning_counts: Counter[str] = Counter()
        for item in qualities:
            warning_counts.update(item.warnings)
        if pose_count == 0:
            recommendation = "No person pose detected. Improve lighting, clean/focus the lens, face the camera and keep head-to-hips in frame."
            status = "pose_not_detected"
        elif accepted_count == 0:
            common = reject_reasons.most_common(1)[0][0] if reject_reasons else "unknown_geometry_failure"
            recommendation = (
                f"MediaPipe detected a person, but required posture geometry was rejected ({common}). "
                "Move back so both shoulders and both hips are visible; keep arms clear of hips. "
                "Do not lower confidence thresholds until landmark overlays have been inspected."
            )
            status = "pose_detected_geometry_rejected"
        elif accepted_count < max(3, int(pose_count * 0.5)):
            recommendation = "Some valid geometry was found, but observation quality is inconsistent. Improve framing/lighting and keep the camera fixed."
            status = "partially_usable"
        else:
            recommendation = "Camera and geometry are usable for a trial session. Validate with a known upright-to-slouch recording."
            status = "usable"
        return {
            "status": status,
            "camera": {
                "width": qualities[-1].width, "height": qualities[-1].height,
                "brightness_mean": mean("brightness_mean"),
                "contrast_std": mean("contrast_std"),
                "sharpness_variance": mean("sharpness_variance"),
                "quality_warning_counts": dict(warning_counts),
            },
            "observations": {
                "frames_read": frame_count, "frames_sampled": sampled_count,
                "image_quality_errors": image_quality_errors,
                "pose_detections": pose_count, "no_pose_detections": no_pose,
                "usable_geometry_samples": accepted_count,
                "geometry_rejection_reasons": dict(reject_reasons),
                "geometry_acceptance_pct": round(100.0 * accepted_count / max(1, len(qualities)), 2),
            },
            "thresholds": {
                "landmark_visibility_min": visibility_min,
                "pose_detection_confidence": getattr(self._detector, "detection_confidence", None),
                "pose_presence_confidence": getattr(self._detector, "presence_confidence", None),
                "pose_tracking_confidence": getattr(self._detector, "tracking_confidence", None),
            },
            "recommendation": recommendation,
        }
