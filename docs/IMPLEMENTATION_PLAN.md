# Implementation Plan

This plan follows the PRD's phases 0 through 9. Work stops after each phase for review and explicit approval before the next begins.

| Phase | Scope | Depends on | Exit evidence |
|---|---|---|---|
| 0 | Repository, Python environment, configuration, logging, documentation | None | Fresh setup instructions; camera-free startup; config tests; safe Git status |
| 1 | Camera adapter, DroidCam setup, webcam and recorded-video sources | 0 | Valid frames and safe cleanup from a documented real source; graceful disconnect |
| 2 | YOLO person detector and debug overlay | 1 | Person boxes/confidence on test footage; documented model requirements |
| 3 | ByteTrack association and temporary IDs | 2 | Tracked objects from recorded input; track limitations documented |
| 4 | Normalized boundary editor, direction, debounce and crossing events | 3 | Direction and duplicate suppression tests; saved/reloaded line |
| 5 | Face detection/capture, quality gate, CLAHE evaluation, optional recognition | 4 | Capture outcome or reason; recognition uncertainty preserved; event survives failures |
| 6 | SQLite, image files, event manager and optional Telegram | 4, 5 | Persistence across restart; remote notification failures isolated |
| 7 | Streamlit dashboard and full pipeline integration | 1–6 | Operator workflow from camera through saved/reviewable event |
| 8 | Reliability, privacy, performance, licence and platform checks | 7 | Test evidence and measured limits; data safeguards documented |
| 9 | Final setup, architecture/workflow diagrams, demo and delivery evidence | 8 | Another developer can reproduce and demonstrate the prototype |

Phase 0 selected Python 3.12 as the initial development baseline because it is available in the current environment. Runtime/model dependency compatibility remains to be checked before those dependencies are committed in their respective phases. Candidate models remain candidates until validated as required by the PRD.

Phase 1 code is on `codex/phase-1-camera`. Mock-capture and generated-video checks passed. The user manually verified moving phone video through OBS DroidCam Virtual Camera at `/dev/video2`. When the phone feed stopped, the preview went blank while OBS kept the virtual device open; the preview utility emitted no disconnect message, so disconnect detection through this specific OBS path remains a known limitation.

Phase 2 implementation is on `codex/phase-2-detection`. Detector and overlay unit tests pass; actual model inference remains unverified because the optional Ultralytics/PyTorch runtime is not installed and the current filesystem has under 1 GB free. The small Ultralytics YOLO11n checkpoint is a prototype candidate, not a production or licensing decision. Its package and trained weights are AGPL-3.0 by default according to Ultralytics; using them in a distributed version may require publishing the complete combined application under AGPL-3.0 or obtaining an Enterprise licence. No model weights will be committed. Runtime performance on the user's computer and model detection on representative footage still require manual verification.
