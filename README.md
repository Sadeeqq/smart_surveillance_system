# Smart Surveillance System

A local-first prototype for detecting and tracking people and creating an event when a tracked person crosses an operator-configured virtual line. Face processing is optional enrichment, and an unknown/no-match result is not evidence of wrongdoing. The project is being built incrementally according to the PRD. Phase 5 adds event-linked face capture, a crop quality gate, optional CLAHE comparison and a replaceable optional recognition adapter. Event database persistence and dashboard integration are later phases.

## Requirements

- Python 3.12 (initial development baseline; model/runtime compatibility will be validated in later phases)
- Git

## Setup

From the repository root, create and activate a virtual environment, then install the phase 0 dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` for local overrides. `.env` is ignored by Git; never put secrets in source files or logs. Missing optional settings use safe defaults.

## Person detection preview (Phase 2)

The detector uses the `yolo11n.pt` pretrained checkpoint as a small prototype candidate and filters detections to the COCO `person` class. On first use, Ultralytics may download the checkpoint to its local cache. Model weights, videos and other local data are not committed. The model runtime adds large PyTorch dependencies; ensure there is adequate free disk space before installing it:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-detection.txt
python -m app.detection_test /dev/video2  # Linux DroidCam Virtual Camera example
python -m app.detection_test path/to/test-video.mp4
```

The first install uses the official CPU-only PyTorch wheel index. Install it before `requirements-detection.txt` so pip does not pull the much larger NVIDIA CUDA packages on a CPU-only setup. The same CPU-only PyTorch command is used by the Phase 2 development environment.

The preview displays person boxes and confidence scores. Press `q` or Esc to close it. `SSS_MODEL_NAME`, `SSS_CONFIDENCE_THRESHOLD`, `SSS_DETECTION_IMAGE_SIZE`, and `SSS_DETECTION_DEVICE` configure the model, minimum score, inference image size, and inference device (`cpu` by default). Do not assume a particular FPS; measure it on the target computer and footage.

## Person tracking preview (Phase 3)

The tracker uses Ultralytics' supported ByteTrack integration with a persistent tracker instance for each source. Install the Phase 2 detection requirements (which include ByteTrack's optional `lap` matching dependency). IDs are temporary, may change after occlusion or missed detections, and must not be treated as identity. The preview draws short bottom-centre trajectories. Run it with a camera or recorded video:

```bash
python -m app.tracking_test /dev/video2 --image-size 320
python -m app.tracking_test path/to/test-video.mp4 --image-size 320
```

Press `q` or Esc to close. `SSS_TRACKING_CONFIDENCE_THRESHOLD` controls the detector threshold supplied to tracking (default `0.1`, allowing ByteTrack to use lower-confidence detections), `SSS_TRACKER_CONFIG` selects an Ultralytics tracker YAML (default `bytetrack.yaml`), and `SSS_TRACKING_TRAIL_LENGTH` controls how many recent points are drawn. The tracking command also accepts `--confidence`, `--tracker`, `--trail-length`, `--image-size`, and `--device` overrides. See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for the Phase 3 manual check.

## Virtual boundary preview and editor (Phase 4)

Phase 4 adds a line editor and direction-aware crossing engine. The endpoints are normalized so the line retains its relative placement when frame dimensions change. Prototype defaults establish a track's initial side over two observations, confirm a crossing over two frames, ignore a neutral zone equal to 1% of the frame diagonal, and apply a five-second per-track cooldown. These are testable starting values, not accuracy guarantees. The editor saves settings per camera under the Git-ignored `data/boundaries.json`, keyed by a SHA-256-derived source ID; the source URL itself is not written to the file. Settings are planned to move into SQLite in Phase 6.

```bash
python -m app.boundary_test /dev/video2 --image-size 256
python -m app.boundary_test path/to/test-video.mp4 --image-size 256
```

Click twice to draw the line. Drag an endpoint to rotate or resize it; drag near the midpoint to move it. Press `1` for A-to-B, `2` for B-to-A, `3` for both directions, `e` to enable/disable, `s` to save, `r` to reset and save the disabled default, and `q` or Esc to quit. A-to-B means movement from the negative to positive signed side of the oriented start-to-end line. Confirmed crossings appear in the terminal as structured event candidates; SQLite event persistence is planned for Phase 6.

## Face capture preview (Phase 5)

The detector uses OpenCV's YuNet adapter. Download the OpenCV Zoo model into the Git-ignored `models/` folder (the model file is not pushed with the repository):

```bash
mkdir -p models
curl -L --fail -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Start the phone feed and configure/enable your saved boundary as in Phase 4. Then run:

```bash
python -m app.boundary_test /dev/video2 --image-size 256 --face-capture
```

For a simple face-detection check with no YOLO, tracking, or boundary logic, run the standalone preview instead:

```bash
python -m app.face_detection_test /dev/video2
```

It draws a box, confidence and five facial landmark points for each detected face. Press `q` or Esc to close. You can also test a recorded video by passing its path instead of `/dev/video2`, and adjust the detector confidence with `--threshold 0.5`.

To preview face detection and person detection/tracking together, without drawing or evaluating a boundary, run:

```bash
python -m app.person_face_tracking_test /dev/video2 --image-size 256
```

The preview draws green person boxes with ByteTrack IDs and orange face boxes with YuNet confidence/landmarks. The top-left status shows current person and face counts plus combined processing FPS. Press `q` or Esc to close. Since this runs YOLO tracking and face detection on every frame, its FPS can be lower than either standalone preview.

When a tracked person crosses the line, the terminal prints the crossing event and a separate face-capture result. Possible results include `captured`, `no_face`, `face_not_in_crossing_track_box`, `low_quality`, and `unavailable`; quality output includes face dimensions, brightness, sharpness indicator and individual checks. For this manual check, use a consenting subject who is close enough for a clear, front-facing image, with even lighting. The crop is processed in memory and is not saved. CLAHE metrics for the original and luminance-enhanced crop are printed for comparison; they are basic image indicators, not identity or accuracy scores.

Recognition is optional and off unless a caller supplies consented enrolled reference images to `DeepFaceFaceNetRecognizer`. The DeepFace package/model weights are not installed or downloaded by default; installing that optional runtime can take substantial disk space and model weights may download on first use. A `known` result is a candidate similarity outcome, not verified identity; results near the threshold remain `inconclusive`. Face thresholds need validation on a labeled, consented set before any accuracy or false-match claims. Do not commit face images or enrolled references.

To try the optional comparison from the boundary preview, first install DeepFace only if you have enough disk space, then pass an explicitly consented reference image stored outside the repository:

```bash
python -m pip install deepface
python -m app.boundary_test /dev/video2 --image-size 256 --face-capture --recognize --enrolled-face "self=/absolute/path/to/consented-reference.jpg"
```

The CLI attempts recognition only after a face passes the capture quality gate. Without references it reports `unknown`; missing DeepFace or model errors report `unavailable`; near-threshold comparisons report `inconclusive`. This is an experimental similarity comparison, not a security decision. DeepFace's wrapper is MIT licensed, but its documentation says underlying model licenses are inherited; review the exact weights before distribution.

The YuNet model is from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and its model directory specifies MIT licensing. DeepFace's wrapper is MIT licensed, but its documentation says underlying model licenses are inherited; review the exact weights before distribution.

**Licensing:** Ultralytics states that its code and trained models are AGPL-3.0 by default. Using them in a distributed application may require licensing the complete combined project under AGPL-3.0 or obtaining an Ultralytics Enterprise licence. Review [Ultralytics licensing](https://www.ultralytics.com/license) before distributing or deploying this integration. The project has not yet selected a final distribution licence.

## Camera preview

The OpenCV adapter accepts a local camera index (`0`), a Linux V4L2 device (`/dev/video0`), an OpenCV-compatible stream URL, or a recorded video file. The preview uses OpenCV's desktop window support:

```bash
python -m app.camera_test 0
python -m app.camera_test /dev/video0
python -m app.camera_test path/to/test-video.mp4
```

Press `q` or Esc to close. Use `SSS_CAMERA_RECONNECT_ATTEMPTS` to configure bounded reconnect attempts for the application adapter.

For the initial DroidCam path, start the phone app and connect the phone and computer to the same network. On Linux, the current DroidCam client exposes the active feed through a V4L2 device; pass the detected `/dev/video*` path or corresponding camera index. On Windows, the client exposes a virtual webcam device that OpenCV can open by the index assigned on that computer. Device discovery and numbering vary, so verify the actual source rather than assuming a universal index or URL. Current vendor setup information: [DroidCam Linux](https://droidcam.app/linux/), [DroidCam Windows](https://droidcam.app/windows/), and [DroidCam help](https://droidcam.app/help/).

The adapter is covered with mocked captures and an automatically generated video file. The user verified live phone video at `/dev/video2` through OBS DroidCam Virtual Camera on their computer; see [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for limits of the manual disconnect check.

## Run and verify

```bash
python -m app.main
pytest
```

The camera adapter and preview utility are available in Phase 1; person detection is available in Phase 2 when its optional runtime is installed. Tracking and boundary crossing preview are available in Phases 3 and 4. Phase 5 face capture is optional and can be tried through the boundary preview. Event persistence, notifications and the dashboard arrive in later phases. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/INTERFACES.md](docs/INTERFACES.md).

## Local data

`data/` and `models/` are excluded from Git. Event image retention, enrolment/removal, access controls and real-deployment policies must be defined before real use; see [docs/DECISIONS.md](docs/DECISIONS.md).
