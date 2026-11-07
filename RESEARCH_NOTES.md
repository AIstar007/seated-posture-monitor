# Research notes — 7 November 2026

## Existing GitHub projects reviewed

- [PostureAI](https://github.com/arefbakali/PostureAI): MediaPipe/OpenCV, personal calibration, head/neck/trunk/shoulder indicators, temporal smoothing, local alerts and dashboard. Borrowed the idea of stable per-user baselines and camera-framing awareness; excluded dashboard, tray UI, notifications and persistence because the take-home explicitly rejects UI/service scope.
- [Ergonomic Posture Monitor](https://github.com/ardaatikk/ergonomic-posture-monitor): pose + face landmarks, five-second calibration, deviation metrics, temporal smoothing, session-level metrics and tests. Borrowed the value of calibration stability checks and auditable measurements; excluded face-landmarker complexity and dashboard.
- [PostureGuard](https://github.com/lukaszkepka/PostureGuard): keypoint classification with a custom training pipeline. Not adopted because the assignment says no trained model.
- [Hunchback Detection](https://github.com/wangyz1999/hunchback-detection): user-labelled posture samples and logistic-regression classification. Useful future direction for threshold validation, but training is outside the time box.
- [Posture Detection](https://github.com/Berohlfs/posture-detection): side-view ear/shoulder/hip geometry and sustained alerts. Useful reminder that anatomical features depend on view.

No source code was copied; these projects informed feature selection and scope decisions.

## Primary technical references

- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker): outputs normalized image landmarks and 3D world landmarks in metres; visibility and presence scores are supplied for landmarks. VIDEO inference is synchronous; LIVE_STREAM is asynchronous and may drop incoming frames while busy. This CLI uses VIDEO mode for ordered, timestamped file/webcam sampling.
- [MediaPipe PyPI release](https://pypi.org/project/mediapipe/1.2.2/): version 1.2.2 was published 3 November 2026. The project pins it and uses the CPU delegate. PyPI reports platform wheels including Windows x86-64 and Linux x86-64.
- [Murugan et al., Ergonomics (2025)](https://pubmed.ncbi.nlm.nih.gov/38293749/): in the compared models/tasks, side-camera placement yielded the lowest MAE. This supports disclosing and fixing the camera view rather than claiming view independence.
- [Aghazadeh & Xiong, Ergonomics (2026)](https://doi.org/10.1016/j.ergon.2026.104061): the reported comparison found that the preferred camera view differed by model; MediaPipe performed best at 45° in their setup. These findings are not contradictory universal rules: camera view is model/task dependent, so the project recommends a fixed 45° side-front starting position and honest setup-specific validation.

## Distinctive design decisions

1. Retain the assignment's CPU-only, headless CLI and no trained model.
2. Use image- and world-landmark visibility/presence; low-confidence observations become unknown, not poor posture.
3. Use scale-independent geometry and disclose camera-view assumptions.
4. Use robust optional calibration with sample count and MAD; exclude setup time from the score and disclose that a personal baseline is not an ideal-posture oracle.
5. Backdate confirmed state transitions and reclassify the candidate interval so break events and good/bad duration totals agree.
6. Report the full-session score together with classified coverage, valid-pose sample rate, confidence, raw feature statistics and self-reported camera view.
7. Keep thresholds as explicit engineering heuristics. Validate with labelled clips before claiming accuracy or clinical relevance.


## V1.2.2 packaging fix

The MediaPipe smoke-test script now adds the repository `src/` directory to `sys.path` before importing the package. This supports direct execution from a fresh checkout without an editable install. The README also documents editable installation for the optional console entry point.


## V1.2.3 calibration reliability follow-up

A Windows webcam run showed that the prior fixed 10-second acquisition timeout could expire before the first usable pose was detected. V1.2.3 separates the wait-for-first-valid-pose window (20 seconds by default) from the five-second baseline sampling period. If observations remain sparse, the collector allows a short three-second sample-completion grace period rather than prematurely scoring uncalibrated observations. Timeout errors now report sampled frames, unusable pose frames, accepted samples and the visibility threshold. The optional preview labels calibration explicitly and displays landmark confidence.

The `--camera-view` option is descriptive metadata only; it does not rotate the camera or select a different geometric feature model.


## V1.2.4: camera diagnostics and rejection transparency

- Separated "no pose detected" from "pose detected but geometry rejected" in processing counters and terminal errors.
- Added per-reason geometry rejection counts (incomplete landmarks, missing/low confidence, invalid coordinates, degenerate torso or shoulder span).
- Added advisory image checks for actual resolution, brightness, contrast and possible blur. These checks do not change posture labels.
- Added a dedicated `--diagnose-camera` use case that reports camera quality, pose detections, valid geometry samples and actionable recommendations without writing a posture session.
- Added optional camera resolution requests and configurable MediaPipe detection/presence/tracking thresholds; actual camera resolution must still be checked because drivers can ignore the request. Defaults remain 0.5 and 1280x720 request to avoid silently weakening inference.
- Preview now overlays the required shoulders/hips and shows detection/geometry rejection messages.
- The diagnostic output is designed to answer whether the issue is camera image quality, person detection, or missing/low-confidence shoulder/hip landmarks.


### Diagnostic interpretation caveat

The camera quality checks are heuristic indicators only. They are reported for troubleshooting and are not a learned image-quality classifier or a gate that decides posture. Geometry validity remains governed by landmark confidence and physically valid feature calculations. Detection thresholds remain 0.5 by default; users can explicitly tune them for experiments, but should compare accepted-geometry rate and false detections before retaining any change.


## Recommended real-camera troubleshooting sequence

1. Run the diagnostic command before a posture session.
2. If no pose is detected, improve lighting/focus and keep the subject's upper body in frame.
3. If pose detections exist but geometry is rejected, inspect the required left/right shoulder and hip overlay and the per-landmark rejection reason.
4. Request 1280x720 capture, then inspect actual reported dimensions; capture backends may ignore resolution requests.
5. Do not reduce MediaPipe or landmark confidence thresholds merely to get a successful exit code. Compare false detections and geometry acceptance on an annotated recording first.


## V1.2.5: diagnostic preview crash fix

A real Windows camera-diagnostic run exposed a `NameError` in the preview landmark renderer: the helper was marked static while referencing the preview instance visibility threshold. The helper is now an instance method, and a regression test verifies required landmarks can be drawn using the configured threshold. This fix affects only optional preview rendering; it does not change detector confidence or posture thresholds.
