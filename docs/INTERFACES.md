# Module Interfaces

These contracts are initial design boundaries. Later phases may add fields where the PRD requires them; changes must be reflected here and in focused tests.

| Module | Input | Output / contract |
|---|---|---|
| Camera source | Device index, V4L2 path, OpenCV-compatible stream URL or recorded video path | `connect()`, `read_frame()`, `status()`, `release()`; OpenCV frames and explicit connection/reconnect/error status |
| Person detector | Non-empty OpenCV frame | Zero or more person detections with frame-relative `xyxy` box, integer model class ID, class label and confidence in `[0, 1]`; raises a clear dependency/model error when its optional runtime is unavailable |
| Tracker | Consecutive frames from one source | Tracked people with temporary ID, confidence, box and bottom-centre anchor; IDs are not identities and may change after lost detections or reset |
| Boundary engine | Tracked anchors and normalized line/direction settings | Crossing candidate with camera, temporary track, boundary, direction and timestamp; no UI dependency |
| Face detector/quality gate | Event context and relevant frames/region | Usable crop or explicit unavailable/low-quality result |
| Image processor | Original face crop | Original retained plus optional CLAHE working crop |
| Recognizer | Usable crop and enrolled references | Known, unknown or inconclusive result, with score/threshold where supported |
| Event manager | Crossing and optional face result | Canonical event plus isolated persistence/notification outcomes |
| Persistence | Event/configuration/image references | SQLite records and local file references; image bytes remain outside SQLite by default |
| Alert adapter | Event summary | Disabled, sent, skipped or failed result; failure cannot erase local event |
| Dashboard | Service interfaces and persisted events | Operator controls, preview/status and event history; no core business logic |

The event record is expected to include event ID, timestamp, camera ID, temporary track ID, boundary ID, direction, detection confidence, face-capture status, recognition status, optional matched identity, image references and alert status. Exact schema is finalized with Phase 6 before implementation.

The Phase 2 detector filters model outputs to the `person` label (case-insensitive) before returning detections. Confidence threshold and image size are configurable. Phase 3 uses Ultralytics' supported `model.track()` API with ByteTrack, which runs detection and association together for consecutive frames. The tracker instance is scoped to one source. Debug annotation draws track IDs and short bottom-centre trajectories on a copy of each frame; IDs are temporary and are not a real-world identity.
