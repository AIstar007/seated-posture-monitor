"""Unit and use-case tests; no real webcam or MediaPipe model is required."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seated_posture_monitor.application import (
    MonitorConfig,
    PoseObservation,
    PostureMonitor,
)
from seated_posture_monitor.domain.geometry import extract_pose_features
from seated_posture_monitor.domain.session import SessionAccumulator
from seated_posture_monitor.domain.temporal import TemporalStabilizer
from seated_posture_monitor.infrastructure.json_reporter import JsonReportWriter
from seated_posture_monitor.infrastructure.preview import OpenCVPreview



def test_preview_draws_required_landmarks_without_name_error():
    """Preview landmark rendering must use the configured visibility threshold."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    preview = OpenCVPreview(visibility_min=0.55)

    preview._draw_required_landmarks(frame, landmarks)

    assert np.count_nonzero(frame) > 0


def pose_landmarks(pitch_deg=4.0, lateral=0.0, roll_deg=2.0, scale=1.0, visibility=0.99):
    """Build 33 MediaPipe-like landmarks in world metres."""
    image = [SimpleNamespace(visibility=visibility, presence=visibility) for _ in range(33)]
    world = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=visibility, presence=visibility) for _ in range(33)]
    vertical = 0.50 * scale
    depth = vertical * math.tan(math.radians(pitch_deg))
    center_x = lateral * vertical
    half_width = 0.18 * scale
    shoulder_dy = math.tan(math.radians(roll_deg)) * half_width
    world[11] = SimpleNamespace(x=center_x - half_width, y=vertical - shoulder_dy / 2, z=depth, visibility=visibility, presence=visibility)
    world[12] = SimpleNamespace(x=center_x + half_width, y=vertical + shoulder_dy / 2, z=depth, visibility=visibility, presence=visibility)
    world[23] = SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=visibility, presence=visibility)
    world[24] = SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=visibility, presence=visibility)
    return image, world


def test_geometry_is_scale_invariant():
    normal = extract_pose_features(*pose_landmarks(scale=1.0))
    closer = extract_pose_features(*pose_landmarks(scale=2.8))
    assert normal is not None and closer is not None
    assert normal.trunk_pitch_deg == pytest.approx(closer.trunk_pitch_deg, abs=0.01)
    assert normal.lateral_lean_ratio == pytest.approx(closer.lateral_lean_ratio, abs=1e-4)
    assert normal.quality_score == pytest.approx(closer.quality_score, abs=0.1)


def test_low_visibility_is_unclassified():
    image, world = pose_landmarks(visibility=0.2)
    assert extract_pose_features(image, world) is None


def test_slouch_scores_lower_than_upright():
    upright = extract_pose_features(*pose_landmarks(pitch_deg=4.0))
    slouched = extract_pose_features(*pose_landmarks(pitch_deg=34.0))
    assert upright is not None and slouched is not None
    assert upright.quality_score > slouched.quality_score


def test_single_bad_frame_does_not_flip_state():
    stabilizer = TemporalStabilizer()
    assert stabilizer.update(0.0, 95.0) is None
    assert stabilizer.update(0.8, 95.0) == "good"
    assert stabilizer.update(0.9, 5.0) == "good"
    assert stabilizer.update(1.0, 95.0) == "good"


def test_sustained_bad_posture_flips_after_confirmation():
    stabilizer = TemporalStabilizer()
    stabilizer.update(0.0, 95.0)
    stabilizer.update(0.8, 95.0)
    assert stabilizer.state == "good"
    for timestamp in (0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1):
        stabilizer.update(timestamp, 5.0)
    assert stabilizer.state == "bad"


def test_session_accumulator_counts_breaks_and_unknown_gaps():
    accumulator = SessionAccumulator(start_s=0.0)
    accumulator.update(1.0, "good", 90.0)
    accumulator.update(2.0, "bad", 40.0)
    accumulator.update(3.0, "good", 90.0)
    accumulator.finish(4.0)
    summary = accumulator.summary(4.0)
    assert summary["posture_break_count"] == 1
    assert summary["bad_posture_duration_s"] == pytest.approx(1.0)
    assert summary["good_posture_duration_s"] == pytest.approx(2.0)
    assert summary["unclassified_duration_s"] == pytest.approx(1.0)


class FakeFrameSource:
    def __init__(self, total=300, fps=10.0, dropped_frame=None):
        self.index = 0
        self.total = total
        self._fps = fps
        self.dropped_frame = dropped_frame
        self.did_drop = False
        self.released = False

    @property
    def is_camera(self):
        return False

    def is_opened(self):
        return True

    def read(self):
        if self.dropped_frame == self.index and not self.did_drop:
            self.index += 1
            self.did_drop = True
            return False, None
        if self.index >= self.total:
            return False, None
        self.index += 1
        return True, np.zeros((24, 24, 3), dtype=np.uint8)

    def fps(self):
        return self._fps

    def frame_count(self):
        return float(self.total)

    def position_seconds(self):
        return self.index / self._fps

    def position_frames(self):
        return float(self.index)

    def seek_frame(self, frame_number):
        self.index = frame_number
        return True

    def release(self):
        self.released = True


class FakePoseDetector:
    def __init__(self, pose_function=None):
        self.pose_function = pose_function or (lambda _time: 4.0)
        self.closed = False

    def detect(self, frame, timestamp_s):
        pitch = self.pose_function(timestamp_s)
        image, world = pose_landmarks(pitch_deg=pitch)
        return PoseObservation(image, world)

    def close(self):
        self.closed = True


class EmptyPoseDetector(FakePoseDetector):
    def detect(self, frame, timestamp_s):
        return None


def run_fake_session(tmp_path, source, detector, duration=30.0, visibility_min=0.55):
    output = tmp_path / "posture.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(
        MonitorConfig(
            source_label="synthetic-test-video.mp4",
            duration_s=duration,
            output_path=output,
            sample_fps=10.0,
            visibility_min=visibility_min,
        )
    )
    return result, output


def test_application_pipeline_writes_summary_and_recovers_dropped_frame(tmp_path):
    source = FakeFrameSource(total=300, dropped_frame=100)
    detector = FakePoseDetector(lambda timestamp: 4.0 if timestamp < 15.0 else 40.0)
    result, output = run_fake_session(tmp_path, source, detector)
    assert result.exit_code == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["metrics"]["posture_break_count"] == 1
    assert summary["metrics"]["bad_posture_duration_s"] > 8.0
    assert summary["processing"]["pose_detected_frames"] >= 290
    assert summary["processing"]["dropped_or_corrupt_frames"] == 1
    assert source.released and detector.closed


def test_application_returns_failure_when_no_person_is_detected(tmp_path):
    source = FakeFrameSource(total=10)
    detector = EmptyPoseDetector()
    result, output = run_fake_session(tmp_path, source, detector, duration=1.0)
    assert result.exit_code == 3
    assert not output.exists()
    assert source.released and detector.closed


class LowConfidencePoseDetector(FakePoseDetector):
    def detect(self, frame, timestamp_s):
        image, world = pose_landmarks(pitch_deg=4.0, visibility=0.2)
        return PoseObservation(image, world)


def test_visibility_cli_setting_is_passed_to_domain_geometry(tmp_path):
    source = FakeFrameSource(total=10)
    detector = LowConfidencePoseDetector()
    result, output = run_fake_session(
        tmp_path, source, detector, duration=1.0, visibility_min=0.1
    )
    assert result.exit_code == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["decision_config"]["landmark_visibility_min"] == pytest.approx(0.1)


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        MonitorConfig(source_label="webcam", duration_s=0)


def test_temporal_transition_exposes_candidate_start_time():
    stabilizer = TemporalStabilizer()
    stabilizer.update(0.0, 95.0)
    stabilizer.update(0.8, 95.0)
    stabilizer.update(1.0, 5.0)
    assert stabilizer.state == "good"
    for timestamp in (1.2, 1.4, 1.6, 1.8, 2.0):
        stabilizer.update(timestamp, 5.0)
    assert stabilizer.state == "bad"
    assert stabilizer.last_transition == ("bad", pytest.approx(1.2))


def test_break_uses_confirmed_candidate_start_not_confirmation_time():
    accumulator = SessionAccumulator(start_s=0.0)
    accumulator.update(1.0, "good", 90.0)
    accumulator.update(2.0, "bad", 40.0, transition_at=1.25)
    accumulator.finish(4.0)
    summary = accumulator.summary(4.0)
    assert summary["posture_breaks"][0]["start_s"] == pytest.approx(1.25)
    assert summary["bad_posture_duration_s"] == pytest.approx(1.75)
    assert summary["good_posture_duration_s"] == pytest.approx(0.25)
    assert summary["posture_breaks"][0]["end_s"] == pytest.approx(3.0)


def test_calibration_uses_median_and_rejects_unstable_samples():
    from seated_posture_monitor.domain.calibration import BaselineCalibrator, CalibrationConfig
    calibrator = BaselineCalibrator(CalibrationConfig(duration_s=2.0, minimum_samples=5))
    for i, pitch in enumerate((4.0, 4.1, 3.9, 4.0, 4.05, 4.0)):
        features = extract_pose_features(*pose_landmarks(pitch_deg=pitch))
        calibrator.add(float(i) * 0.5, features)
    baseline = calibrator.finish()
    assert baseline.trunk_pitch_deg == pytest.approx(4.0, abs=0.2)
    assert baseline.samples == 6

    unstable = BaselineCalibrator(CalibrationConfig(duration_s=2.0, minimum_samples=5))
    for i, pitch in enumerate((0.0, 35.0, 2.0, 38.0, 1.0, 40.0)):
        unstable.add(float(i) * 0.5, extract_pose_features(*pose_landmarks(pitch_deg=pitch)))
    with pytest.raises(ValueError, match="unstable"):
        unstable.finish()


def test_calibration_requires_minimum_observations():
    from seated_posture_monitor.domain.calibration import BaselineCalibrator, CalibrationConfig
    calibrator = BaselineCalibrator(CalibrationConfig(duration_s=1.0, minimum_samples=5))
    features = extract_pose_features(*pose_landmarks())
    calibrator.add(0.0, features)
    with pytest.raises(ValueError, match="at least"):
        calibrator.finish()


def test_geometry_observation_confidence_is_reported():
    image, world = pose_landmarks(visibility=0.85)
    features = extract_pose_features(image, world)
    assert features is not None
    assert features.observation_confidence == pytest.approx(0.85)


def test_parse_source_keeps_numeric_video_filename_as_string():
    from seated_posture_monitor.cli import parse_source
    assert parse_source("0") == 0
    assert parse_source("12345.mp4") == "12345.mp4"


def test_calibrated_session_excludes_setup_time_and_uses_full_monitor_duration(tmp_path):
    source = FakeFrameSource(total=300, fps=10.0)
    detector = FakePoseDetector(lambda _timestamp: 4.0)
    output = tmp_path / "calibrated.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(MonitorConfig(
        source_label="calibration-test.mp4",
        duration_s=10.0,
        output_path=output,
        sample_fps=10.0,
        calibration_enabled=True,
        calibration_duration_s=5.0,
    ))
    assert result.exit_code == 0, result.message
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["session"]["calibration"]["status"] == "completed"
    assert report["session"]["calibration_duration_excluded_from_score_s"] >= 5.0
    assert report["metrics"]["session_duration_s"] == pytest.approx(10.0, abs=0.2)
    assert report["metrics"]["stable_observation_coverage_pct"] > 80.0


def test_backdated_transition_reclassifies_durations_and_break_end():
    accumulator = SessionAccumulator(start_s=0.0)
    accumulator.update(1.0, "good", 95.0, observation_confidence=0.99)
    accumulator.update(1.2, "good", 75.0, observation_confidence=0.98)
    accumulator.update(1.4, "good", 65.0, observation_confidence=0.98)
    accumulator.update(1.6, "good", 55.0, observation_confidence=0.97)
    accumulator.update(2.0, "bad", 45.0, transition_at=1.2, observation_confidence=0.96)
    accumulator.update(2.2, "bad", 40.0, observation_confidence=0.96)
    accumulator.update(2.6, "good", 80.0, transition_at=2.2, observation_confidence=0.99)
    accumulator.finish(4.0)

    metrics = accumulator.summary(4.0)
    assert metrics["good_posture_duration_s"] == pytest.approx(1.6)
    assert metrics["bad_posture_duration_s"] == pytest.approx(1.0)
    assert metrics["unclassified_duration_s"] == pytest.approx(1.4)
    assert metrics["posture_breaks"] == [
        {"start_s": 1.2, "end_s": 2.2, "duration_s": 1.0}
    ]
    assert metrics["mean_landmark_confidence"] == pytest.approx(0.98, abs=0.02)


def test_calibration_acquisition_timeout_reports_actionable_diagnostics(tmp_path):
    source = FakeFrameSource(total=300, fps=10.0)
    detector = EmptyPoseDetector()
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(MonitorConfig(
        source_label="no-pose.mp4",
        duration_s=10.0,
        output_path=tmp_path / "no-pose.json",
        calibration_enabled=True,
        calibration_duration_s=5.0,
        calibration_acquisition_timeout_s=2.0,
    ))
    assert result.exit_code == 3
    assert "No usable pose features were accepted" in result.message
    assert "accepted_calibration_samples=0" in result.message
    assert "unusable_pose_frames=" in result.message
    assert not (tmp_path / "no-pose.json").exists()


def test_calibration_can_begin_after_delayed_first_pose(tmp_path):
    class DelayedPoseDetector(FakePoseDetector):
        def detect(self, frame, timestamp_s):
            if timestamp_s < 4.0:
                return None
            return super().detect(frame, timestamp_s)

    source = FakeFrameSource(total=300, fps=10.0)
    detector = DelayedPoseDetector()
    output = tmp_path / "delayed-calibration.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(MonitorConfig(
        source_label="delayed-pose.mp4",
        duration_s=5.0,
        output_path=output,
        calibration_enabled=True,
        calibration_duration_s=5.0,
        calibration_acquisition_timeout_s=6.0,
    ))
    assert result.exit_code == 0, result.message
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["session"]["calibration"]["status"] == "completed"
    assert report["session"]["calibration"]["baseline"]["samples"] >= 20


def test_calibration_waits_for_sample_count_after_collection_window(tmp_path):
    class SparsePoseDetector(FakePoseDetector):
        def detect(self, frame, timestamp_s):
            if int(timestamp_s * 10) % 5 != 0:
                return None
            return super().detect(frame, timestamp_s)

    source = FakeFrameSource(total=300, fps=10.0)
    detector = SparsePoseDetector()
    output = tmp_path / "sparse-calibration.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(MonitorConfig(
        source_label="sparse-pose.mp4",
        duration_s=5.0,
        output_path=output,
        calibration_enabled=True,
        calibration_duration_s=5.0,
        calibration_acquisition_timeout_s=3.0,
    ))
    assert result.exit_code == 3
    assert "requires at least" in result.message
    assert "usable samples" in result.message
    assert not output.exists()


def test_calibration_enabled_but_source_ends_early_is_failure(tmp_path):
    source = FakeFrameSource(total=20, fps=10.0)
    detector = FakePoseDetector()
    output = tmp_path / "incomplete-calibration.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(
        MonitorConfig(
            source_label="short-calibration.mp4",
            duration_s=10.0,
            output_path=output,
            calibration_enabled=True,
            calibration_duration_s=5.0,
        )
    )
    assert result.exit_code != 0
    assert not output.exists()


def test_confidence_uses_world_and_image_landmark_confidence():
    image, world = pose_landmarks(visibility=0.95)
    world[11].visibility = 0.2
    assert extract_pose_features(image, world) is None


def test_invalid_frame_channel_shape_is_rejected():
    assert not PostureMonitor._is_valid_frame(np.zeros((24, 24, 1), dtype=np.uint8))
    assert PostureMonitor._is_valid_frame(np.zeros((24, 24, 3), dtype=np.uint8))


class FailingCameraSource(FakeFrameSource):
    @property
    def is_camera(self):
        return True

    def read(self):
        if self.index >= 15:
            return False, None
        return super().read()


def test_repeated_webcam_read_failure_is_nonzero_and_writes_no_report(tmp_path):
    source = FailingCameraSource(total=100, fps=10.0)
    detector = FakePoseDetector()
    output = tmp_path / "failed-camera.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(
        MonitorConfig(source_label="webcam-0", duration_s=10.0, output_path=output)
    )
    assert result.exit_code != 0
    assert "Webcam capture failed" in result.message
    assert not output.exists()
    assert source.released and detector.closed


def test_summary_reports_feature_statistics_and_observation_confidence():
    accumulator = SessionAccumulator(start_s=0.0)
    f1 = extract_pose_features(*pose_landmarks(pitch_deg=4.0, visibility=0.9))
    f2 = extract_pose_features(*pose_landmarks(pitch_deg=8.0, visibility=0.8))
    accumulator.record_features(f1)
    accumulator.record_features(f2)
    accumulator.update(0.5, "good", 95.0, observation_confidence=f1.observation_confidence)
    accumulator.update(1.0, "good", 90.0, observation_confidence=f2.observation_confidence)
    accumulator.finish(1.5)
    metrics = accumulator.summary(1.5)
    assert metrics["feature_statistics"]["trunk_pitch_deg"]["samples"] == 2
    assert metrics["feature_statistics"]["trunk_pitch_deg"]["mean"] == pytest.approx(6.0, abs=0.1)
    assert metrics["mean_landmark_confidence"] == pytest.approx(0.85, abs=0.01)


def test_calibration_minimum_samples_adapts_to_low_sampling_rate(tmp_path):
    source = FakeFrameSource(total=200, fps=10.0)
    detector = FakePoseDetector()
    output = tmp_path / "low-rate-calibration.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(
        MonitorConfig(
            source_label="low-rate.mp4",
            duration_s=5.0,
            output_path=output,
            sample_fps=2.0,
            calibration_enabled=True,
            calibration_duration_s=5.0,
        )
    )
    assert result.exit_code == 0, result.message
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["session"]["calibration"]["stability_limits"]["minimum_samples"] == 5


def test_missing_confidence_metadata_is_not_assumed_to_be_perfect():
    image, world = pose_landmarks(visibility=0.95)
    for landmark in (image[11], world[11]):
        landmark.visibility = None
        landmark.presence = None
    assert extract_pose_features(image, world) is None


def test_camera_view_is_validated_and_reported(tmp_path):
    with pytest.raises(ValueError, match="camera_view"):
        MonitorConfig(source_label="video", camera_view="top-down")
    source = FakeFrameSource(total=40, fps=10.0)
    detector = FakePoseDetector()
    output = tmp_path / "view.json"
    monitor = PostureMonitor(
        frame_source=source,
        pose_detector=detector,
        report_writer=JsonReportWriter(),
        utc_now=lambda: datetime(2026, 11, 7, tzinfo=timezone.utc),
        monotonic=lambda: 100.0,
    )
    result = monitor.run(
        MonitorConfig(
            source_label="view-test.mp4",
            duration_s=3.0,
            output_path=output,
            camera_view="45-degree",
        )
    )
    assert result.exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["session"]["camera_view"] == "45-degree"


def test_default_model_cache_path_is_versioned():
    from seated_posture_monitor.infrastructure.model_asset import MODEL_FILENAME, MODEL_URL, default_model_path
    assert "-v1" in MODEL_FILENAME
    assert "/float16/1/" in MODEL_URL
    assert default_model_path().name == MODEL_FILENAME


def test_recoverable_inference_error_is_counted_and_session_continues(tmp_path):
    from seated_posture_monitor.application.ports import RecoverableFrameError

    class OneFrameFailureDetector(FakePoseDetector):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def detect(self, frame, timestamp_s):
            self.calls += 1
            if self.calls == 1:
                raise RecoverableFrameError("synthetic single-frame inference failure")
            return super().detect(frame, timestamp_s)

    detector = OneFrameFailureDetector()
    result, output = run_fake_session(
        tmp_path,
        FakeFrameSource(total=80, fps=10.0),
        detector,
        duration=5.0,
    )

    assert result.exit_code == 0, result.message
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["processing"]["recoverable_inference_errors"] == 1
    assert report["processing"]["pose_detected_frames"] >= 5
    assert any("recoverable inference errors" in warning for warning in result.warnings)


# V1.2.4 camera-diagnostics and rejection-reason regression tests.
def test_geometry_analysis_explains_low_confidence_landmark():
    from seated_posture_monitor.domain.geometry import analyze_pose_features
    image, world = pose_landmarks(visibility=0.95)
    image[23].visibility = 0.10
    result = analyze_pose_features(image, world)
    assert result.features is None
    assert result.reason is not None
    assert "left_hip_image_visibility_low" in result.reason


def test_geometry_analysis_distinguishes_missing_confidence_metadata():
    from seated_posture_monitor.domain.geometry import analyze_pose_features
    image, world = pose_landmarks(visibility=0.95)
    image[11].visibility = None
    image[11].presence = None
    result = analyze_pose_features(image, world)
    assert result.features is None
    assert "left_shoulder_image_confidence_metadata_missing" in result.reason


def test_camera_quality_analyzer_flags_dark_blurry_low_resolution_frame():
    from seated_posture_monitor.infrastructure.camera_quality import CameraQualityAnalyzer
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = CameraQualityAnalyzer().analyze(frame)
    assert result.width == 320 and result.height == 240
    assert "low_resolution" in result.warnings
    assert "low_light" in result.warnings
    assert "possible_blur_or_soft_focus" in result.warnings


def test_camera_diagnostic_separates_no_pose_from_rejected_geometry():
    from seated_posture_monitor.application.camera_diagnostics import CameraDiagnosticUseCase
    from seated_posture_monitor.infrastructure.camera_quality import CameraQualityAnalyzer

    source = FakeFrameSource(total=40, fps=10.0)
    detector = LowConfidencePoseDetector()
    class FastClock:
        def __init__(self): self.value = 0.0
        def __call__(self):
            self.value += 0.05
            return self.value
    result = CameraDiagnosticUseCase(
        source, detector, quality_analyzer=CameraQualityAnalyzer(), monotonic=FastClock()
    ).run(duration_s=1.0, sample_fps=5.0, visibility_min=0.55)
    assert result["status"] == "pose_detected_geometry_rejected"
    assert result["observations"]["pose_detections"] > 0
    assert result["observations"]["usable_geometry_samples"] == 0
    assert any("low_or_missing_landmark_confidence" in reason for reason in result["observations"]["geometry_rejection_reasons"])


def test_camera_diagnostic_identifies_no_person():
    from seated_posture_monitor.application.camera_diagnostics import CameraDiagnosticUseCase
    from seated_posture_monitor.infrastructure.camera_quality import CameraQualityAnalyzer
    source = FakeFrameSource(total=40, fps=10.0)
    result = CameraDiagnosticUseCase(
        source, EmptyPoseDetector(), quality_analyzer=CameraQualityAnalyzer()
    ).run(duration_s=1.0, sample_fps=5.0)
    assert result["status"] == "pose_not_detected"
    assert "No person pose detected" in result["recommendation"]


def test_monitor_failure_message_contains_specific_landmark_rejection(tmp_path):
    source = FakeFrameSource(total=40, fps=10.0)
    detector = LowConfidencePoseDetector()
    result, output = run_fake_session(tmp_path, source, detector, duration=3.0)
    assert result.exit_code == 3
    assert "geometry was rejected" in result.message
    assert "left_shoulder_image_visibility_low" in result.message
    assert not output.exists()

def test_cli_exposes_camera_diagnostic_and_resolution_arguments():
    from seated_posture_monitor.cli import build_parser
    options = build_parser().parse_args([
        "--source", "0", "--diagnose-camera", "--width", "1280",
        "--height", "720", "--pose-detection-confidence", "0.4",
    ])
    assert options.diagnose_camera is True
    assert options.width == 1280 and options.height == 720
    assert options.pose_detection_confidence == pytest.approx(0.4)

def test_detector_confidence_thresholds_are_validated():
    from seated_posture_monitor.application.config import MonitorConfig
    with pytest.raises(ValueError, match="pose_detection_confidence"):
        MonitorConfig(source_label="webcam", pose_detection_confidence=1.1)
