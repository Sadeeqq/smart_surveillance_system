# Smart Surveillance System

A local-first prototype for detecting and tracking people and creating an event when a tracked person crosses an operator-configured virtual line. Face processing is optional enrichment, and an unknown/no-match result is not evidence of wrongdoing. The project is being built incrementally according to the PRD. Phase 2 adds a YOLO person detector and debug preview; tracking and later pipeline stages have not been implemented yet.

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

The camera adapter and preview utility are available in Phase 1; person detection is available in Phase 2 when its optional runtime is installed. Tracking, boundary events, face processing, persistence, notifications and the dashboard arrive in later phases. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/INTERFACES.md](docs/INTERFACES.md).

## Local data

`data/` and `models/` are excluded from Git. Event image retention, enrolment/removal, access controls and real-deployment policies must be defined before real use; see [docs/DECISIONS.md](docs/DECISIONS.md).
