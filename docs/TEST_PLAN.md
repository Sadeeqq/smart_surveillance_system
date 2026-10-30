# Test Plan

Tests use pytest as specified in the PRD. Hardware-dependent checks will use mocks or recorded footage where possible; a physical camera is required only for the documented camera compatibility demonstration.

| Test layer | Planned coverage | Phase |
|---|---|---|
| Unit | Configuration defaults/validation; detector person filtering, confidence, boxes, labels and annotation; tracker IDs, filtering, anchors and trajectory overlay; boundary geometry, direction, jitter and re-arm; event schema; storage helpers | 0, 2, 3, 4, 6 |
| Integration | Mock/recorded frames through camera and detector, tracker and boundary; crossing through face attempt/event manager/SQLite; optional services failing independently | 1–7 |
| End-to-end | Camera to visible, stored, reviewable event; restart and restore history/configuration | 7 |
| Failure injection | Camera loss, model load/inference errors, track loss, no/poor face, DB/image write failures, Telegram offline/disabled | 1–8 |
| Manual evaluation | Person detection on recorded footage, lighting, angles, occlusion, multiple people, authorized face samples, CLAHE comparison, FPS/latency/CPU/RAM | 2–8 |

For each phase, record exact commands, pass/fail/skip counts, the environment, and unverified manual scenarios. Never treat startup alone as acceptance. Camera adapter tests use fake captures and a generated recorded-video source; at least one real camera configuration still requires a manual check. Configuration tests cover safe defaults, invalid values and local `.env` precedence.

Phase 2 must verify actual model inference on test footage separately from mocked detector unit tests. Record model/checkpoint source, runtime versions, CPU/GPU choice, confidence and image-size settings, and observed FPS. Do not commit downloaded weights or sample footage containing identifiable people.

## Phase 1 verification record

- Environment: Python 3.12.3, OpenCV 4.14.0, Linux; no `/dev/video*` device is exposed to this workspace.
- Automated: `.venv/bin/python -m pytest -q` — 9 passed, 0 failed, 0 skipped.
- Startup: `.venv/bin/python -m app.main` — passed.
- Preview CLI: `.venv/bin/python -m app.camera_test --help` — passed.
- Recorded source: generated MJPG video — valid 32×24 OpenCV frame read successfully in the automated suite.
- Real camera probe in this workspace: `.venv/bin/python -m app.camera_test 0` — failed as expected because OpenCV reports no camera device at index 0.
- User manual check on Linux: DroidCam phone feed shown as OBS `/dev/video2` named `DroidCam Virtual Camera`; `.venv/bin/python -m app.camera_test /dev/video2` showed a moving live phone image. User stopped the phone feed and observed a blank image with no terminal message because OBS kept the loopback device producing frames. Live-camera success is confirmed; disconnect-error detection through this OBS configuration remains unverified and is documented as a limitation.

## Phase 2 verification record (implementation review)

- Environment: Python 3.12.3, Linux x86_64, OpenCV 4.14.0, Ultralytics 8.4.192, PyTorch 2.15.0+cpu, CPU inference. The detector runtime was installed in the project virtual environment, and `models/yolo11n.pt` was downloaded from the Ultralytics release assets. `models/` is Git-ignored.
- Automated: `.venv/bin/python -m pytest -q` — 16 passed, 0 failed, 0 skipped. Detector behavior is covered with a fake model (person filtering, confidence, coordinates/labels, configurable inference options, error propagation and non-mutating overlay); these tests do not claim actual YOLO inference.
- CLI: `.venv/bin/python -m app.detection_test --help` — passed.
- Actual inference smoke check: Ultralytics sample `bus.jpg`, 810×1080, `yolo11n.pt`, CPU, confidence 0.5, image size 640 — 4 person boxes returned (scores 0.888, 0.878, 0.856, 0.622). This confirms model loading and real inference on a public sample image, not the live phone camera or a recorded clip.
- User manual live-camera check: DroidCam source `/dev/video2`, pretrained `yolo11n.pt`, CPU, confidence threshold 0.5, image size 320 — one person detected at score 0.78 and approximately 2.5 FPS. This confirms a basic live person-detection path, not precision/recall or model accuracy.
- User manual command with the phone/OBS virtual camera running: `.venv/bin/python -m app.detection_test /dev/video2`. Point the phone at a clearly visible person and confirm a box plus confidence score appears; press `q` to exit. For recorded footage, use `.venv/bin/python -m app.detection_test path/to/test-video.mp4`. Record model, device, confidence, image size and observed FPS before accepting Phase 2.

## Phase 3 verification record (implementation review)

- Tracker: Ultralytics `model.track()` with the ByteTrack configuration; `persist=True` is used only for consecutive frames from one source. ByteTrack's `lap` matching dependency is included in `requirements-detection.txt`.
- Automated tests cover tracked-person parsing, temporary IDs, bottom-centre anchors, tracker arguments/errors, configuration validation, and trajectory annotation with fake model outputs. These tests do not claim real ByteTrack association quality.
- Real-runtime smoke check: bundled Ultralytics `bus.jpg`, pretrained `yolo11n.pt`, CPU, image size 320, two repeated frames — ByteTrack returned four person IDs (1–4) on both calls. This confirms the installed ByteTrack dependency and wrapper/API integration; repeated still frames do not establish movement or occlusion stability. PyTorch emitted non-fatal NNPACK unsupported-hardware warnings on this Atom CPU.
- User manual live-camera check: DroidCam virtual-camera source `/dev/video2`, with its virtual-camera output active, CPU, image size 256 — user reports ByteTrack follows the moving person with the same temporary track number. FPS remained slow; no exact value was reported at 256. This confirms basic live ID continuity, not performance or accuracy guarantees.
- The recorded-video preview remains available for repeatable checks: `.venv/bin/python -m app.tracking_test path/to/test-video.mp4`. IDs may change after occlusion, missed detections or stream restart; they are not identities. Additional multi-person, occlusion and recorded-footage checks remain for later reliability evaluation.

## Phase 4 verification record (implementation review)

- Geometry: normalized endpoints are mapped to actual frame dimensions; signed-side calculations are tested at multiple frame sizes.
- State logic: tests cover initial appearance (no event), enabled/disabled direction, disabled line, neutral-zone jitter, a one-frame side change, confirmed crossing, reverse crossing/re-arm, per-track cooldown, event fields and setting validation.
- Settings: JSON save/reload is tested per camera. Camera IDs are SHA-256-derived and raw source URLs are not stored. The file is local-only until migration to SQLite in Phase 6.
- Automated: `.venv/bin/python -m pytest -q` — 36 passed, 0 failed, 0 skipped. This includes fake-observation tests for the crossing engine and mouse-editor coordinate tests; it does not validate live camera/tracker performance.
- CLI: `.venv/bin/python -m app.boundary_test --help` — passed.
- Manual (user-reported): the live boundary editor, saving and reloading the line, and the direction-aware crossing check all worked with the user's DroidCam setup at `/dev/video2`. The user reports the preview and controls work; no exact FPS or separate jitter benchmark was reported. Other cameras and footage remain unverified.
