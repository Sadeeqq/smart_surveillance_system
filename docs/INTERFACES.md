# Module Interfaces

These contracts are initial design boundaries. Later phases may add fields where the PRD requires them; changes must be reflected here and in focused tests.

| Module | Input | Output / contract |
|---|---|---|
| Camera source | Device index, V4L2 path, OpenCV-compatible stream URL or recorded video path | `connect()`, `read_frame()`, `status()`, `release()`; OpenCV frames and explicit connection/reconnect/error status |
| Person detector | Frame | Zero or more detections with frame-relative box, class label and confidence |
| Tracker | Detections plus prior tracker state | Tracked people with temporary ID, box and bottom-centre anchor |
| Boundary engine | Tracked anchors and normalized line/direction settings | Crossing candidate with camera, temporary track, boundary, direction and timestamp; no UI dependency |
| Face detector/quality gate | Event context and relevant frames/region | Usable crop or explicit unavailable/low-quality result |
| Image processor | Original face crop | Original retained plus optional CLAHE working crop |
| Recognizer | Usable crop and enrolled references | Known, unknown or inconclusive result, with score/threshold where supported |
| Event manager | Crossing and optional face result | Canonical event plus isolated persistence/notification outcomes |
| Persistence | Event/configuration/image references | SQLite records and local file references; image bytes remain outside SQLite by default |
| Alert adapter | Event summary | Disabled, sent, skipped or failed result; failure cannot erase local event |
| Dashboard | Service interfaces and persisted events | Operator controls, preview/status and event history; no core business logic |

The event record is expected to include event ID, timestamp, camera ID, temporary track ID, boundary ID, direction, detection confidence, face-capture status, recognition status, optional matched identity, image references and alert status. Exact schema is finalized with Phase 6 before implementation.
