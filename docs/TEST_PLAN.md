# Test Plan

Tests use pytest as specified in the PRD. Hardware-dependent checks will use mocks or recorded footage where possible; a physical camera is required only for the documented camera compatibility demonstration.

| Test layer | Planned coverage | Phase |
|---|---|---|
| Unit | Configuration defaults/validation; boundary geometry, direction, jitter and re-arm; event schema; storage helpers | 0, 4, 6 |
| Integration | Mock/recorded frames through detector, tracker and boundary; crossing through face attempt/event manager/SQLite; optional services failing independently | 1–7 |
| End-to-end | Camera to visible, stored, reviewable event; restart and restore history/configuration | 7 |
| Failure injection | Camera loss, model load/inference errors, track loss, no/poor face, DB/image write failures, Telegram offline/disabled | 1–8 |
| Manual evaluation | Lighting, angles, occlusion, multiple people, authorized face samples, CLAHE comparison, FPS/latency/CPU/RAM | 2–8 |

For each phase, record exact commands, pass/fail/skip counts, the environment, and unverified manual scenarios. Never treat startup alone as acceptance. Phase 0's current automated suite covers safe defaults and invalid confidence values; no camera/model checks are expected yet.
