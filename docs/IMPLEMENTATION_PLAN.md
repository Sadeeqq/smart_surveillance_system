# Implementation Plan

This plan follows the PRD's phases 0 through 9. Work stops after each phase for review and explicit approval before the next begins.

| Phase | Scope | Depends on | Exit evidence |
|---|---|---|---|
| 0 | Repository, Python environment, configuration, logging, documentation | None | Fresh setup instructions; camera-free startup; config tests; safe Git status |
| 1 | Camera adapter, DroidCam setup, webcam and recorded-video sources | 0 | Valid frames and safe cleanup from a documented real source; graceful disconnect |
| 2 | YOLO person detector and debug overlay | 1 | Person boxes/confidence on test footage; documented model requirements |
| 3 | ByteTrack association, temporary IDs, trajectory preview | 2 | Tracked objects from recorded input; track limitations documented |
| 4 | Normalized boundary editor, direction, debounce and crossing events | 3 | Direction and duplicate suppression tests; saved/reloaded line |
| 5 | Face detection/capture, quality gate, CLAHE evaluation, optional recognition | 4 | Capture outcome or reason; recognition uncertainty preserved; event survives failures |
| 6 | SQLite, image files, event manager and optional Telegram | 4, 5 | Persistence across restart; remote notification failures isolated |
| 7 | Streamlit dashboard and full pipeline integration | 1–6 | Operator workflow from camera through saved/reviewable event |
| 8 | Reliability, privacy, performance, licence and platform checks | 7 | Test evidence and measured limits; data safeguards documented |
| 9 | Final setup, architecture/workflow diagrams, demo and delivery evidence | 8 | Another developer can reproduce and demonstrate the prototype |

Phase 0 selected Python 3.12 as the initial development baseline because it is available in the current environment. Runtime/model dependency compatibility remains to be checked before those dependencies are committed in their respective phases. Candidate models remain candidates until validated as required by the PRD.

Phase 1 code is on `codex/phase-1-camera`. Mock-capture and generated-video checks passed. The user manually verified moving phone video through OBS DroidCam Virtual Camera at `/dev/video2`. When the phone feed stopped, the preview went blank while OBS kept the virtual device open; the preview utility emitted no disconnect message, so disconnect detection through this specific OBS path remains a known limitation.

Phase 2 implementation is on `codex/phase-2-detection`. Detector and overlay unit tests pass. The user manually verified person detection on the live DroidCam source `/dev/video2` at image size 320, observing a person score of 0.78 and 2.5 FPS on CPU. This is a smoke check, not an accuracy evaluation. Model weights remain local and are downloaded by Ultralytics when needed. The small Ultralytics YOLO11n checkpoint is a prototype candidate, not a production or licensing decision. Its package and trained weights are AGPL-3.0 by default according to Ultralytics; review licensing before distribution.

Phase 3 work is on `codex/phase-3-tracking`. Tracking uses the supported Ultralytics `model.track()` API with the ByteTrack configuration and persistent state scoped to one source. The optional `lap` matching dependency is listed with the detection runtime. The user manually confirmed the live DroidCam feed tracks a moving person with a mostly consistent temporary ID at image size 256. Processing is still slow on the Intel Atom test PC; no exact FPS was recorded for the 256 setting. IDs can change after missed detections, occlusion or tracker reset.

Phase 4 work is on `codex/phase-4-boundary`. It adds normalized per-camera line settings, a UI-independent signed-side/debounce engine, structured crossing candidates and an OpenCV editor/preview CLI. The editor saves to local JSON under `data/`, keyed by a hash of the source string so stream credentials are not written. Prototype defaults require two initial-side observations, two opposite-side frames, a neutral zone of 1% of frame diagonal and a five-second per-track cooldown. Automated tests cover geometry, direction filtering, appearance-without-crossing, jitter/one-frame debounce, cooldown and settings round-trip. The user manually confirmed the live Phase 4 editor, saving/reloading the line and direction-aware crossing test worked on their DroidCam setup. Exact FPS and threshold suitability on other footage remain unmeasured.

Phase 5 is implemented on `codex/phase-5-face-processing`. It adds an OpenCV YuNet adapter, event-track face association, explicit crop quality outcomes, original-preserving luminance CLAHE variants and an optional DeepFace/FaceNet recognizer adapter. The YuNet model and any enrollment images stay local and outside Git. Recognition is not enabled by default; unknown or inconclusive results never alter the crossing event or imply wrongdoing. Automated checks passed (42 total). The user manually verified two live DroidCam boundary crossings with face capture; both crops passed all quality checks. This does not test identity recognition. The candidate's false-match rate, performance and underlying model licenses require evaluation before deployment.
