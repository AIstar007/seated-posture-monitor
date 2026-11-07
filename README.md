# Seated Posture Monitor

**Version 1.2.5** · Python 3.10+ · CPU-only · single-person CLI

Tracks one seated person's upper-body landmarks from a webcam or video and writes an auditable JSON session summary. It is an engineering demonstration, not a medical or ergonomic diagnosis.

## Run (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python posture_monitor.py --source 0 --duration 45 --out posture.json
```

For a video file, replace `0` with a path. Add `--preview` for an optional window. The official MediaPipe Lite model is downloaded once to the user's cache on first use. Run `python scripts/smoke_test_mediapipe.py` to verify model loading and CPU inference.

## Diagnose a camera before recording

```powershell
python posture_monitor.py --source 0 --diagnose-camera --preview --diagnostic-seconds 10 --width 1280 --height 720
```

The diagnostic separates image-quality warnings (resolution, brightness, contrast and possible blur), MediaPipe pose detections, and geometry rejection reasons. Requested camera resolution may be ignored by the driver; inspect the actual dimensions in the report. Image-quality thresholds are advisory and never determine posture. Preview overlays the required shoulders/hips and their visibility values. MediaPipe detection, presence and tracking thresholds default to `0.5`; they can be tuned explicitly, but lowering them can increase false detections. Do not weaken thresholds simply to obtain a successful run.

## Decisions

MediaPipe world landmarks provide scale-independent trunk pitch, torso-normalized lateral offset and shoulder roll. Both shoulders and both hips must have image- and world-landmark visibility/presence of at least **0.55**; otherwise the sample is **unknown**, not poor posture. Fixed-mode good/bad bands are **10°/25°** trunk pitch, **0.05/0.18** lateral ratio and **5°/18°** shoulder roll. Scores interpolate linearly between each band and combine with weights **65%/20%/15%**. These are transparent engineering heuristics, not clinically validated thresholds. A **0.65 s EMA**, **58/72 hysteresis** and **0.75 s confirmation** suppress frame flicker; transitions use the filtered candidate start time. The session score counts unknown time against the full session rather than treating missing observations as good posture.

Optional `--calibrate` collects a five-second stable personal baseline. It is relative to the posture held during calibration, not proof of ideal posture; setup time is excluded from scoring. `--camera-view` records user-declared metadata only and does not transform geometry.

## Preview regression fix (V1.2.5)

The camera-diagnostic preview now uses the configured landmark-visibility threshold through its instance state. A regression test exercises landmark rendering to prevent a `NameError` when `--preview` is enabled.

## Architecture, testing and limitations

Clean Architecture: CLI composition root → application use cases/ports → domain rules; OpenCV and MediaPipe adapters implement outer dependencies. Responsibilities are separated for capture, inference, geometry, calibration, temporal stabilization, aggregation and JSON reporting.

```powershell
python -m pytest -q
python -m compileall -q posture_monitor.py src tests scripts
```

Tests use synthetic landmarks and fake adapters; they validate software behavior, not real-world posture accuracy. Before submission, record **30–60 seconds** with an upright phase followed by a slouched phase and replace the clearly marked illustrative `posture.json` with the genuine output. Camera angle, lighting, focus and framing affect landmark quality. The current geometry requires both shoulders and both hips; it does not handle multiple people, lower-body posture, chair/desk ergonomics or medical conditions. Keep the head, shoulders and hips visible and keep the camera fixed.
