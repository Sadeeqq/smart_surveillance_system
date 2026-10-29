# Test Plan

Tests use pytest as specified in the PRD. Hardware-dependent checks will use mocks or recorded footage where possible; a physical camera is required only for the documented camera compatibility demonstration.

| Test layer | Planned coverage | Phase |
|---|---|---|
| Unit | Configuration defaults/validation; detector person filtering, confidence, boxes, labels and annotation; boundary geometry, direction, jitter and re-arm; event schema; storage helpers | 0, 2, 4, 6 |
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

- Environment: Python 3.12 virtual environment, OpenCV installed, Linux; Ultralytics is not installed. The workspace filesystem has under 1 GB free, so the PyTorch-based optional runtime and downloaded weights were not installed here.
- Automated: `.venv/bin/python -m pytest -q` — 16 passed, 0 failed, 0 skipped. Detector behavior is covered with a fake model (person filtering, confidence, coordinates/labels, configurable inference options, error propagation and non-mutating overlay); these tests do not claim actual YOLO inference.
- CLI: `.venv/bin/python -m app.detection_test --help` — passed.
- Not yet verified: pretrained model load/download, person boxes on actual test footage, target-device FPS/latency and model licence choice for distribution.
- Manual command after installing the optional runtime and ensuring sufficient disk space: `python -m pip install -r requirements-detection.txt`, then `python -m app.detection_test /dev/video2` or `python -m app.detection_test path/to/test-video.mp4`. Record the model, device, confidence, image size and observed FPS before accepting Phase 2.
