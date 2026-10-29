# Test Plan

Tests use pytest as specified in the PRD. Hardware-dependent checks will use mocks or recorded footage where possible; a physical camera is required only for the documented camera compatibility demonstration.

| Test layer | Planned coverage | Phase |
|---|---|---|
| Unit | Configuration defaults/validation; boundary geometry, direction, jitter and re-arm; event schema; storage helpers | 0, 4, 6 |
| Integration | Mock/recorded frames through detector, tracker and boundary; crossing through face attempt/event manager/SQLite; optional services failing independently | 1–7 |
| End-to-end | Camera to visible, stored, reviewable event; restart and restore history/configuration | 7 |
| Failure injection | Camera loss, model load/inference errors, track loss, no/poor face, DB/image write failures, Telegram offline/disabled | 1–8 |
| Manual evaluation | Lighting, angles, occlusion, multiple people, authorized face samples, CLAHE comparison, FPS/latency/CPU/RAM | 2–8 |

For each phase, record exact commands, pass/fail/skip counts, the environment, and unverified manual scenarios. Never treat startup alone as acceptance. Camera adapter tests use fake captures and a generated recorded-video source; at least one real camera configuration still requires a manual check. Configuration tests cover safe defaults, invalid values and local `.env` precedence.

## Phase 1 verification record

- Environment: Python 3.12.3, OpenCV 4.14.0, Linux; no `/dev/video*` device is exposed to this workspace.
- Automated: `.venv/bin/python -m pytest -q` — 9 passed, 0 failed, 0 skipped.
- Startup: `.venv/bin/python -m app.main` — passed.
- Preview CLI: `.venv/bin/python -m app.camera_test --help` — passed.
- Recorded source: generated MJPG video — valid 32×24 OpenCV frame read successfully in the automated suite.
- Real camera probe: `.venv/bin/python -m app.camera_test 0` — failed as expected because OpenCV reports no camera device at index 0; this does not satisfy the real camera acceptance criterion.
- GUI preview and DroidCam are unverified. Use `python -m app.camera_test /dev/videoN` on Linux or the actual OpenCV camera index on Windows after the DroidCam client exposes its virtual camera.
