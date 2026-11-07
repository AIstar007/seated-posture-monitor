"""Application use case: coordinate capture, inference, domain rules and output."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from seated_posture_monitor.domain.geometry import DEFAULT_THRESHOLDS, analyze_pose_features, extract_pose_features
from seated_posture_monitor.domain.calibration import BaselineCalibrator, CalibrationConfig, CalibrationBaseline
from seated_posture_monitor.domain.models import (
    GeometryThresholds,
    PoseFeatures,
    RunCounters,
    StabilizerConfig,
)
from seated_posture_monitor.domain.session import SessionAccumulator
from seated_posture_monitor.infrastructure.camera_quality import CameraQualityAnalyzer
from seated_posture_monitor.domain.temporal import TemporalStabilizer

from .config import MonitorConfig
from .ports import FrameSource, PoseDetector, Preview, ReportWriter, RecoverableFrameError

from seated_posture_monitor import __version__ as APP_VERSION
MIN_POSE_DETECTIONS = 5


@dataclass(frozen=True)
class MonitorResult:
    exit_code: int
    summary: Optional[dict[str, Any]] = None
    message: Optional[str] = None
    warnings: tuple[str, ...] = ()


class PostureMonitor:
    """Orchestrate a session through injected ports (Dependency Inversion).

    This class owns the workflow, not OpenCV/MediaPipe details. Each dependency
    is replaceable with another adapter or a fake in a test.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        pose_detector: PoseDetector,
        report_writer: ReportWriter,
        preview: Optional[Preview] = None,
        feature_extractor: Callable[..., Optional[PoseFeatures]] = extract_pose_features,
        geometry_thresholds: GeometryThresholds = DEFAULT_THRESHOLDS,
        stabilizer_config: StabilizerConfig = StabilizerConfig(),
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._frame_source = frame_source
        self._pose_detector = pose_detector
        self._report_writer = report_writer
        self._preview = preview
        self._feature_extractor = feature_extractor
        self._geometry_thresholds = geometry_thresholds
        self._stabilizer_config = stabilizer_config
        self._monotonic = monotonic
        self._utc_now = utc_now
        self._camera_quality = CameraQualityAnalyzer()

    def run(self, config: MonitorConfig) -> MonitorResult:
        source = self._frame_source
        detector = self._pose_detector
        warnings: list[str] = []
        if not source.is_opened():
            self._close_resources()
            return MonitorResult(2, message=f"Could not open video source: {config.source_label}")

        geometry_thresholds = replace(
            self._geometry_thresholds, visibility_min=config.visibility_min
        )
        fps = source.fps()
        if not math.isfinite(fps) or fps <= 1.0 or fps > 240.0:
            fps = 30.0

        started_at = self._utc_now().isoformat(timespec="seconds")
        start_monotonic = self._monotonic()
        counters = RunCounters()
        stabilizer = TemporalStabilizer(self._stabilizer_config)
        accumulator = SessionAccumulator(
            start_s=0.0, max_gap_s=self._stabilizer_config.max_observation_gap_s
        )
        previous_timestamp = -1.0
        last_sample_timestamp = -1.0
        last_timestamp = 0.0
        first_pose_timestamp: Optional[float] = None
        last_pose_timestamp: Optional[float] = None
        frame_index = 0
        consecutive_read_failures = 0
        end_reason = "eof"
        video_frame_count = 0.0
        expected_calibration_samples = config.sample_fps * config.calibration_duration_s
        calibration_config = CalibrationConfig(
            duration_s=config.calibration_duration_s,
            minimum_samples=max(5, min(20, int(expected_calibration_samples * 0.4))),
        )
        calibrator = BaselineCalibrator(calibration_config) if config.calibration_enabled else None
        calibration_baseline: Optional[CalibrationBaseline] = None
        calibration_started_at: Optional[float] = None
        session_origin_s = 0.0
        calibration_timed_out = False
        calibration_timeout_detail = ""
        camera_read_failed = False
        frame_error_examples: list[str] = []

        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    if (
                        source.is_camera
                        and (calibrator is None or calibration_baseline is not None)
                        and self._monotonic() - start_monotonic >= session_origin_s + config.duration_s
                    ):
                        end_reason = "duration_limit"
                        break
                    consecutive_read_failures += 1
                    if source.is_camera:
                        counters.dropped_or_corrupt_frames += 1
                        if consecutive_read_failures >= 3:
                            camera_read_failed = True
                            end_reason = "camera_read_failure"
                            warnings.append(
                                "Webcam read failed three consecutive times; the session is incomplete."
                            )
                            break
                        continue

                    # Recover a damaged file frame when decoder position metadata
                    # confirms there are still frames available to skip to.
                    try:
                        next_frame = source.position_frames()
                        total_frames = source.frame_count()
                    except (TypeError, ValueError, OSError):
                        next_frame, total_frames = 0.0, 0.0
                    if (
                        math.isfinite(next_frame)
                        and math.isfinite(total_frames)
                        and total_frames > 0
                        and next_frame < total_frames
                        and consecutive_read_failures <= 5
                    ):
                        counters.dropped_or_corrupt_frames += 1
                        target_frame = max(frame_index + 1, int(round(next_frame)))
                        if source.seek_frame(target_frame):
                            frame_index = target_frame - 1
                            continue
                    break

                consecutive_read_failures = 0
                counters.frames_read += 1
                frame_index += 1
                if not self._is_valid_frame(frame):
                    counters.dropped_or_corrupt_frames += 1
                    continue

                timestamp = self._frame_timestamp(
                    source, start_monotonic, frame_index, fps, previous_timestamp
                )
                previous_timestamp = timestamp
                if calibrator is None or calibration_baseline is not None:
                    if timestamp - session_origin_s >= config.duration_s:
                        last_timestamp = session_origin_s + config.duration_s
                        end_reason = "duration_limit"
                        break
                elif calibration_started_at is None and timestamp > config.calibration_acquisition_timeout_s:
                    calibration_timed_out = True
                    calibration_timeout_detail = (
                        f"No usable pose features were accepted within "
                        f"{config.calibration_acquisition_timeout_s:.1f}s. "
                        f"sampled_frames={counters.frames_sampled}, "
                        f"unusable_pose_frames={counters.frames_without_pose}, "
                        f"accepted_calibration_samples={calibrator.sample_count}, "
                        f"visibility_threshold={config.visibility_min:.2f}. "
                        "Keep both shoulders and both hips visible, move farther from the camera, "
                        "and avoid arms or objects covering the torso."
                    )
                    end_reason = "calibration_timeout"
                    break
                last_timestamp = timestamp

                min_sample_delta = 1.0 / max(config.sample_fps, 0.1)
                if (
                    last_sample_timestamp >= 0
                    and timestamp - last_sample_timestamp + 1e-9 < min_sample_delta
                ):
                    continue
                last_sample_timestamp = timestamp
                counters.frames_sampled += 1

                try:
                    observation = detector.detect(frame, timestamp)
                    if observation is None:
                        counters.frames_without_pose += 1
                        counters.frames_pose_not_detected += 1
                        stabilizer.note_missing(timestamp)
                        image_quality = self._safe_frame_quality(frame)
                        if self._show_preview(frame, None, None, None, "NO PERSON POSE DETECTED", None, image_quality):
                            end_reason = "user_quit"
                            break
                        continue

                    counters.pose_model_detected_frames += 1
                    if self._feature_extractor is extract_pose_features:
                        analysis = analyze_pose_features(
                            observation.image_landmarks,
                            observation.world_landmarks,
                            geometry_thresholds,
                        )
                        features = analysis.features
                        rejection_reason = analysis.reason
                    else:
                        features = self._feature_extractor(
                            observation.image_landmarks,
                            observation.world_landmarks,
                            geometry_thresholds,
                        )
                        rejection_reason = None if features is not None else "custom_feature_extractor_rejected"
                    if features is None:
                        counters.frames_without_pose += 1
                        counters.frames_geometry_rejected += 1
                        reason_key = (rejection_reason or "geometry_rejected").split(":", 1)[0]
                        counters.geometry_rejection_reasons[reason_key] = counters.geometry_rejection_reasons.get(reason_key, 0) + 1
                        if rejection_reason and len(counters.geometry_rejection_examples) < 5:
                            counters.geometry_rejection_examples.append(rejection_reason)
                        stabilizer.note_missing(timestamp)
                        image_quality = self._safe_frame_quality(frame)
                        detail = f"LANDMARKS REJECTED: {rejection_reason or 'unknown geometry reason'}"
                        if self._show_preview(frame, None, None, None, detail, observation, image_quality):
                            end_reason = "user_quit"
                            break
                        continue

                    counters.pose_detected_frames += 1
                    if calibrator is not None and calibration_baseline is None:
                        calibrator.add(timestamp, features)
                        if calibration_started_at is None:
                            calibration_started_at = timestamp
                        calibration_elapsed = timestamp - calibration_started_at
                        minimum_span = config.calibration_duration_s * 0.8
                        if calibration_elapsed >= config.calibration_duration_s:
                            enough_samples = calibrator.sample_count >= calibration_config.minimum_samples
                            enough_span = calibrator.sample_span_s >= minimum_span
                            if enough_samples and enough_span:
                                try:
                                    calibration_baseline = calibrator.finish()
                                except ValueError as exc:
                                    return MonitorResult(
                                        3,
                                        message=(
                                            f"Calibration failed: {exc}. "
                                            f"Accepted samples={calibrator.sample_count}; "
                                            f"sample span={calibrator.sample_span_s:.2f}s."
                                        ),
                                        warnings=tuple(warnings),
                                    )
                                warnings.append("Personal baseline calibration completed; scoring is relative to this segment, not a clinical ideal.")
                                # Begin a fresh scoring clock after calibration. Setup time
                                # is not included in the requested monitoring duration or score.
                                session_origin_s = timestamp
                                accumulator = SessionAccumulator(
                                    start_s=0.0,
                                    max_gap_s=self._stabilizer_config.max_observation_gap_s,
                                )
                                stabilizer.reset()
                                first_pose_timestamp = None
                                last_pose_timestamp = None
                                continue
                            if calibration_elapsed >= config.calibration_duration_s + 3.0:
                                calibration_timed_out = True
                                calibration_timeout_detail = (
                                    f"Calibration collected {calibrator.sample_count} usable samples "
                                    f"over {calibrator.sample_span_s:.2f}s; requires at least "
                                    f"{calibration_config.minimum_samples} samples spanning "
                                    f"{minimum_span:.2f}s. "
                                    f"sampled_frames={counters.frames_sampled}, "
                                    f"pose_not_detected_frames={counters.frames_pose_not_detected}, "
                                    f"geometry_rejected_frames={counters.frames_geometry_rejected}, "
                                    f"geometry_rejection_reasons={counters.geometry_rejection_reasons}, "
                                    f"visibility_threshold={config.visibility_min:.2f}. "
                                    "Keep both shoulders and hips visible and hold a comfortable, "
                                    "steady seated position."
                                )
                                end_reason = "calibration_timeout"
                                break
                        if calibration_baseline is None:
                            if self._show_preview(frame, "calibrating", features.quality_score, features, "COLLECTING BASELINE", observation, self._safe_frame_quality(frame)):
                                end_reason = "user_quit"
                                break
                            continue
                    if calibrator is not None and calibration_baseline is not None:
                        # Adjust only the baseline-relative components. The absolute metric remains in the report.
                        pitch_delta = abs(features.trunk_pitch_deg - calibration_baseline.trunk_pitch_deg)
                        lateral_delta = abs(features.lateral_lean_ratio - calibration_baseline.lateral_lean_ratio)
                        roll_delta = abs(features.shoulder_roll_deg - calibration_baseline.shoulder_roll_deg)
                        pitch_score = max(
                            0.0,
                            100.0 * (1.0 - pitch_delta / calibration_config.pitch_zero_score_delta_deg),
                        )
                        lateral_score = max(
                            0.0,
                            100.0 * (1.0 - lateral_delta / calibration_config.lateral_zero_score_delta_ratio),
                        )
                        roll_score = max(
                            0.0,
                            100.0 * (1.0 - roll_delta / calibration_config.shoulder_roll_zero_score_delta_deg),
                        )
                        adjusted_score = (
                            geometry_thresholds.pitch_weight * pitch_score
                            + geometry_thresholds.lateral_weight * lateral_score
                            + geometry_thresholds.shoulder_weight * roll_score
                        )
                        features = replace(features, quality_score=round(adjusted_score, 2))
                    session_timestamp = max(0.0, timestamp - session_origin_s)
                    accumulator.record_features(features)
                    if first_pose_timestamp is None:
                        first_pose_timestamp = session_timestamp
                    last_pose_timestamp = session_timestamp
                    stable_state = stabilizer.update(session_timestamp, features.quality_score)
                    transition = stabilizer.last_transition
                    transition_at = transition[1] if transition else None
                    accumulator.update(
                        session_timestamp,
                        stable_state,
                        features.quality_score,
                        transition_at,
                        features.observation_confidence,
                    )
                    if self._show_preview(frame, stable_state, features.quality_score, features, "POSE GEOMETRY VALID", observation, self._safe_frame_quality(frame)):
                        end_reason = "user_quit"
                        break
                except RecoverableFrameError as exc:
                    counters.dropped_or_corrupt_frames += 1
                    counters.recoverable_inference_errors += 1
                    stabilizer.note_missing(timestamp)
                    if len(frame_error_examples) < 3:
                        frame_error_examples.append(str(exc))
                    continue
        except Exception as exc:
            return MonitorResult(4, message=f"Session processing failed: {exc}", warnings=tuple(warnings))
        finally:
            # Read metadata before release; some capture backends clear it on release.
            if not source.is_camera:
                try:
                    video_frame_count = source.frame_count()
                except Exception:
                    video_frame_count = 0.0
            # Lifecycle cleanup is centralised for EOF, errors and preview quit.
            self._close_resources()

        elapsed = max(0.0, self._monotonic() - start_monotonic)
        if source.is_camera:
            end_timestamp = min(elapsed, session_origin_s + config.duration_s) if end_reason != "eof" else elapsed
        elif end_reason == "duration_limit":
            end_timestamp = session_origin_s + config.duration_s
        elif end_reason == "user_quit":
            end_timestamp = last_timestamp + 1.0 / fps
        else:
            estimated_end = video_frame_count / fps if video_frame_count > 0 else last_timestamp + 1.0 / fps
            if not math.isfinite(estimated_end) or estimated_end <= 0:
                estimated_end = last_timestamp + 1.0 / fps
            end_timestamp = min(estimated_end, session_origin_s + config.duration_s)
        end_timestamp = max(session_origin_s, min(end_timestamp, session_origin_s + config.duration_s))

        scored_end_timestamp = max(0.0, end_timestamp - session_origin_s)
        accumulator.finish(scored_end_timestamp)
        if frame_error_examples:
            examples = "; ".join(frame_error_examples)
            warnings.append(
                f"Skipped {counters.recoverable_inference_errors} frame(s) after recoverable inference errors. "
                f"Examples: {examples}"
            )
        pose_span = (
            last_pose_timestamp - first_pose_timestamp
            if first_pose_timestamp is not None and last_pose_timestamp is not None
            else 0.0
        )
        if camera_read_failed:
            return MonitorResult(4, message="Webcam capture failed repeatedly; no complete session report was written.", warnings=tuple(warnings))
        if calibration_timed_out:
            return MonitorResult(
                3,
                message=f"Calibration timed out. {calibration_timeout_detail}",
                warnings=tuple(warnings),
            )
        if calibrator is not None and calibration_baseline is None:
            return MonitorResult(3, message="Calibration did not complete; no scored session was produced.", warnings=tuple(warnings))
        if counters.pose_detected_frames < MIN_POSE_DETECTIONS or pose_span < 0.25:
            if counters.pose_model_detected_frames == 0:
                detail = (
                    "MediaPipe did not detect a person in sampled frames. Check lighting, focus, "
                    "camera framing and that the subject is facing the camera."
                )
            elif counters.frames_geometry_rejected > 0:
                common_reason = max(
                    counters.geometry_rejection_reasons,
                    key=counters.geometry_rejection_reasons.get,
                )
                example = (
                    counters.geometry_rejection_examples[0]
                    if counters.geometry_rejection_examples else common_reason
                )
                detail = (
                    f"MediaPipe detected a person, but posture geometry was rejected: {common_reason}. "
                    f"Example: {example}. Keep both shoulders and both hips visible; move the camera farther away and "
                    "keep arms/desk clear of the hip area. See --diagnose-camera for a breakdown."
                )
            else:
                detail = "Too few usable posture observations were collected."
            return MonitorResult(3, message=detail, warnings=tuple(warnings))

        summary = self._build_summary(
            config, started_at, scored_end_timestamp, accumulator, counters, source.is_camera,
            geometry_thresholds, calibration_baseline, session_origin_s, calibration_config
        )
        try:
            self._report_writer.write(config.output_path, summary)
        except (OSError, ValueError, TypeError) as exc:
            return MonitorResult(4, message=f"Could not write report: {exc}", warnings=tuple(warnings))
        return MonitorResult(0, summary=summary, warnings=tuple(warnings))

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        # Avoid importing NumPy in the application layer. Only non-empty BGR/BGRA
        # arrays are accepted by the OpenCV/MediaPipe adapters.
        if frame is None or getattr(frame, "ndim", 0) != 3 or getattr(frame, "size", 0) <= 0:
            return False
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3:
            return False
        height, width, channels = shape
        return height > 0 and width > 0 and channels in (3, 4)

    def _safe_frame_quality(self, frame: Any) -> Any:
        if self._preview is None:
            return None
        try:
            return self._camera_quality.analyze(frame)
        except (TypeError, ValueError, AttributeError):
            return None

    def _show_preview(
        self, frame: Any, state: Optional[str], quality: Optional[float], features: Any,
        status_detail: Optional[str] = None, observation: Any = None, image_quality: Any = None,
    ) -> bool:
        if not self._preview:
            return False
        return self._preview.show(
            frame, state, quality, features, status_detail, observation, image_quality
        )

    def _frame_timestamp(
        self,
        source: FrameSource,
        start_monotonic: float,
        frame_index: int,
        fps: float,
        previous_timestamp: float,
    ) -> float:
        if source.is_camera:
            current = self._monotonic() - start_monotonic
        else:
            pos_ms = source.position_seconds()
            current = pos_ms if math.isfinite(pos_ms) and pos_ms > 0 else 0.0
            if current <= previous_timestamp:
                current = frame_index / fps
        if current <= previous_timestamp:
            current = previous_timestamp + 1.0 / fps
        return current

    def _build_summary(
        self,
        config: MonitorConfig,
        started_at: str,
        duration_s: float,
        accumulator: SessionAccumulator,
        counters: RunCounters,
        is_camera: bool,
        geometry_thresholds: GeometryThresholds,
        calibration_baseline: Optional[CalibrationBaseline] = None,
        calibration_excluded_s: float = 0.0,
        calibration_config: CalibrationConfig = CalibrationConfig(),
    ) -> dict[str, Any]:
        geometry = geometry_thresholds
        stabilizer = self._stabilizer_config
        return {
            "schema_version": "1.2",
            "application": {"name": "seated-posture-monitor", "version": APP_VERSION},
            "session": {
                "started_at_utc": started_at,
                "source_type": "webcam" if is_camera else "video_file",
                "source": config.source_label,
                "camera_view": config.camera_view,
                "preview_enabled": config.preview_enabled,
                "calibration_duration_excluded_from_score_s": round(calibration_excluded_s, 2),
                "calibration": {
                    "enabled": config.calibration_enabled,
                    "status": "completed" if calibration_baseline else "disabled" if not config.calibration_enabled else "failed_or_incomplete",
                    "baseline": ({
                        "trunk_pitch_deg": round(calibration_baseline.trunk_pitch_deg, 2),
                        "lateral_lean_ratio": round(calibration_baseline.lateral_lean_ratio, 4),
                        "shoulder_roll_deg": round(calibration_baseline.shoulder_roll_deg, 2),
                        "samples": calibration_baseline.samples,
                        "duration_s": round(calibration_baseline.duration_s, 2),
                        "median_absolute_deviation": {
                            "trunk_pitch_deg": round(calibration_baseline.pitch_mad_deg, 3),
                            "lateral_lean_ratio": round(calibration_baseline.lateral_mad_ratio, 4),
                            "shoulder_roll_deg": round(calibration_baseline.shoulder_roll_mad_deg, 3),
                        },
                    } if calibration_baseline else None),
                    "stability_limits": {
                        "minimum_samples": calibration_config.minimum_samples,
                        "maximum_pitch_mad_deg": calibration_config.maximum_pitch_mad_deg,
                        "maximum_lateral_mad_ratio": calibration_config.maximum_lateral_mad_ratio,
                        "maximum_shoulder_roll_mad_deg": calibration_config.maximum_roll_mad_deg,
                    } if config.calibration_enabled else None,
                    "relative_score_zero_delta": {
                        "pitch_deg": calibration_config.pitch_zero_score_delta_deg,
                        "lateral_ratio": calibration_config.lateral_zero_score_delta_ratio,
                        "shoulder_roll_deg": calibration_config.shoulder_roll_zero_score_delta_deg,
                    } if config.calibration_enabled else None,
                },
            },
            "metrics": accumulator.summary(duration_s),
            "processing": {
                "frames_read": counters.frames_read,
                "frames_sampled": counters.frames_sampled,
                "pose_detected_frames": counters.pose_detected_frames,
                "frames_without_pose": counters.frames_without_pose,
                "frames_pose_not_detected": counters.frames_pose_not_detected,
                "pose_model_detected_frames": counters.pose_model_detected_frames,
                "frames_geometry_rejected": counters.frames_geometry_rejected,
                "geometry_rejection_reasons": dict(counters.geometry_rejection_reasons),
                "geometry_rejection_examples": list(counters.geometry_rejection_examples),
                "dropped_or_corrupt_frames": counters.dropped_or_corrupt_frames,
                "recoverable_inference_errors": counters.recoverable_inference_errors,
                "valid_pose_sample_pct": round(
                    100.0 * counters.pose_detected_frames / counters.frames_sampled, 2
                ) if counters.frames_sampled else 0.0,
            },
            "decision_config": {
                "landmark_visibility_min": geometry.visibility_min,
                "trunk_pitch_good_deg": geometry.pitch_good_deg,
                "trunk_pitch_bad_deg": geometry.pitch_bad_deg,
                "lateral_lean_good_ratio": geometry.lateral_good_ratio,
                "lateral_lean_bad_ratio": geometry.lateral_bad_ratio,
                "shoulder_roll_good_deg": geometry.shoulder_roll_good_deg,
                "shoulder_roll_bad_deg": geometry.shoulder_roll_bad_deg,
                "feature_weights": {
                    "trunk_pitch": geometry.pitch_weight,
                    "lateral_lean": geometry.lateral_weight,
                    "shoulder_roll": geometry.shoulder_weight,
                },
                "ema_time_constant_s": stabilizer.ema_time_constant_s,
                "bad_enter_score": stabilizer.bad_enter_score,
                "good_enter_score": stabilizer.good_enter_score,
                "state_confirmation_s": stabilizer.state_confirmation_s,
                "max_observation_gap_s": stabilizer.max_observation_gap_s,
                "sample_fps": config.sample_fps,
                "calibration_duration_s": config.calibration_duration_s if config.calibration_enabled else None,
                "calibration_acquisition_timeout_s": config.calibration_acquisition_timeout_s if config.calibration_enabled else None,
                "pose_model": config.model_name,
                "inference_delegate": "CPU",
                "pose_detection_confidence": config.pose_detection_confidence,
                "pose_presence_confidence": config.pose_presence_confidence,
                "pose_tracking_confidence": config.pose_tracking_confidence,
            },
        }

    def _close_resources(self) -> None:
        # Release each owned dependency explicitly; adapters remain responsible
        # for their own native handles. Cleanup must not mask the run result.
        for resource, method_name in (
            (self._preview, "close"),
            (self._pose_detector, "close"),
            (self._frame_source, "release"),
        ):
            if resource is None:
                continue
            try:
                getattr(resource, method_name)()
            except Exception:
                pass
