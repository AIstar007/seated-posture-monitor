"""Command-line interface and dependency-composition root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import cv2

from seated_posture_monitor.application import MonitorConfig, PostureMonitor
from seated_posture_monitor.application.camera_diagnostics import CameraDiagnosticUseCase
from seated_posture_monitor.infrastructure import (
    JsonReportWriter,
    MediaPipePoseDetector,
    OpenCVPreview,
    OpenCVVideoSource,
    ensure_model,
    CameraQualityAnalyzer,
)


def parse_source(value: str):
    """Convert a numeric CLI source to a webcam index; otherwise keep its path."""
    source_text = value.strip()
    try:
        return int(source_text)
    except ValueError:
        return source_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track one seated person's upper-body posture from a webcam or video "
            "and write a JSON session summary."
        )
    )
    parser.add_argument("--source", default="0", help="Webcam index (0) or video-file path")
    parser.add_argument("--duration", type=float, default=60.0, help="Maximum session length in seconds")
    parser.add_argument("--out", default="posture.json", help="Output JSON path")
    parser.add_argument("--model", default=None, help="Optional path to a downloaded .task model file")
    parser.add_argument("--visibility", type=float, default=0.55, help="Minimum landmark visibility [0-1]")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Maximum inference rate (default: 10)")
    parser.add_argument("--preview", action="store_true", help="Show optional preview; headless by default")
    parser.add_argument("--calibrate", action="store_true", help="Collect a short neutral baseline before scoring")
    parser.add_argument("--calibration-seconds", type=float, default=5.0, help="Neutral baseline collection duration (default: 5s)")
    parser.add_argument("--calibration-timeout-seconds", type=float, default=20.0, help="Maximum wait for the first usable pose during calibration (default: 20s)")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width (default: 1280; driver may ignore)")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height (default: 720; driver may ignore)")
    parser.add_argument("--pose-detection-confidence", type=float, default=0.5, help="MediaPipe pose detection confidence [0-1]")
    parser.add_argument("--pose-presence-confidence", type=float, default=0.5, help="MediaPipe pose presence confidence [0-1]")
    parser.add_argument("--pose-tracking-confidence", type=float, default=0.5, help="MediaPipe tracking confidence [0-1]")
    parser.add_argument("--diagnose-camera", action="store_true", help="Diagnose camera image quality, pose detection and geometry")
    parser.add_argument("--diagnostic-seconds", type=float, default=8.0, help="Duration of camera diagnostics (default: 8s)")
    parser.add_argument(
        "--camera-view",
        choices=("unspecified", "frontal", "45-degree", "side"),
        default="unspecified",
        help="Self-reported camera angle for the session report",
    )
    return parser


def run_monitor(options: argparse.Namespace) -> int:
    source = None
    detector = None
    preview = None
    try:
        source_value = parse_source(options.source)
        source = OpenCVVideoSource(
            source_value,
            width=options.width if isinstance(source_value, int) else None,
            height=options.height if isinstance(source_value, int) else None,
        )
        if not source.is_opened():
            print(f"ERROR: Could not open video source: {options.source}", file=sys.stderr)
            source.release()
            return 2

        model_path = ensure_model(Path(options.model) if options.model else None)
        detector = MediaPipePoseDetector(
            model_path,
            detection_confidence=options.pose_detection_confidence,
            presence_confidence=options.pose_presence_confidence,
            tracking_confidence=options.pose_tracking_confidence,
        )
        if options.diagnose_camera:
            diagnostic_preview = OpenCVPreview(visibility_min=options.visibility) if options.preview else None
            diagnostic = CameraDiagnosticUseCase(
                source, detector, quality_analyzer=CameraQualityAnalyzer(), preview=diagnostic_preview
            ).run(
                duration_s=options.diagnostic_seconds,
                sample_fps=options.sample_fps,
                visibility_min=options.visibility,
            )
            import json
            print(json.dumps(diagnostic, indent=2))
            return 0 if diagnostic.get("status") == "usable" else 3
        preview = OpenCVPreview(visibility_min=options.visibility) if options.preview else None
        config = MonitorConfig(
            source_label=str(options.source),
            duration_s=options.duration,
            output_path=Path(options.out),
            visibility_min=options.visibility,
            sample_fps=options.sample_fps,
            preview_enabled=bool(options.preview),
            model_name=model_path.name,
            calibration_enabled=bool(options.calibrate),
            calibration_duration_s=options.calibration_seconds,
            calibration_acquisition_timeout_s=options.calibration_timeout_seconds,
            camera_view=options.camera_view,
            pose_detection_confidence=options.pose_detection_confidence,
            pose_presence_confidence=options.pose_presence_confidence,
            pose_tracking_confidence=options.pose_tracking_confidence,
        )
        monitor = PostureMonitor(
            frame_source=source,
            pose_detector=detector,
            report_writer=JsonReportWriter(),
            preview=preview,
        )
        result = monitor.run(config)
        for warning in result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if result.exit_code != 0:
            print(f"ERROR: {result.message or 'Posture monitoring failed'}", file=sys.stderr)
            return result.exit_code

        assert result.summary is not None
        metrics = result.summary["metrics"]
        print(
            f"Saved {options.out} | score {metrics['posture_score_0_100']:.1f}/100 | "
            f"breaks {metrics['posture_break_count']} | "
            f"good posture {metrics['good_posture_pct']:.1f}%"
        )
        return 0
    except KeyboardInterrupt:
        print("Interrupted before a complete session could be written.", file=sys.stderr)
        return 130
    except (cv2.error, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: Could not initialise posture monitoring: {exc}", file=sys.stderr)
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass
        if preview is not None:
            try:
                preview.close()
            except Exception:
                pass
        if source is not None:
            try:
                source.release()
            except Exception:
                pass
        return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    if options.duration <= 0:
        parser.error("--duration must be greater than 0")
    if not 0.0 <= options.visibility <= 1.0:
        parser.error("--visibility must be between 0 and 1")
    if options.sample_fps <= 0 or options.sample_fps > 60:
        parser.error("--sample-fps must be greater than 0 and at most 60")
    if options.calibration_seconds <= 0:
        parser.error("--calibration-seconds must be greater than 0")
    if options.calibration_timeout_seconds <= 0:
        parser.error("--calibration-timeout-seconds must be greater than 0")
    if options.diagnostic_seconds <= 0:
        parser.error("--diagnostic-seconds must be greater than 0")
    if options.width < 1 or options.height < 1:
        parser.error("--width and --height must be positive")
    for argument in (options.pose_detection_confidence, options.pose_presence_confidence, options.pose_tracking_confidence):
        if not 0.0 <= argument <= 1.0:
            parser.error("MediaPipe confidence options must be between 0 and 1")
    return run_monitor(options)
