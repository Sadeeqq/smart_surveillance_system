# Smart Surveillance System

A local-first prototype for detecting and tracking people and creating an event when a tracked person crosses an operator-configured virtual line. Face processing is optional enrichment, and an unknown/no-match result is not evidence of wrongdoing. The project is being built incrementally according to the PRD. Phase 1 adds a camera adapter and preview utility; person detection and later pipeline stages have not been implemented yet.

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

## Camera preview

The OpenCV adapter accepts a local camera index (`0`), a Linux V4L2 device (`/dev/video0`), an OpenCV-compatible stream URL, or a recorded video file. The preview uses OpenCV's desktop window support:

```bash
python -m app.camera_test 0
python -m app.camera_test /dev/video0
python -m app.camera_test path/to/test-video.mp4
```

Press `q` or Esc to close. Use `SSS_CAMERA_RECONNECT_ATTEMPTS` to configure bounded reconnect attempts for the application adapter.

For the initial DroidCam path, start the phone app and connect the phone and computer to the same network. On Linux, the current DroidCam client exposes the active feed through a V4L2 device; pass the detected `/dev/video*` path or corresponding camera index. On Windows, the client exposes a virtual webcam device that OpenCV can open by the index assigned on that computer. Device discovery and numbering vary, so verify the actual source rather than assuming a universal index or URL. Current vendor setup information: [DroidCam Linux](https://droidcam.app/linux/), [DroidCam Windows](https://droidcam.app/windows/), and [DroidCam help](https://droidcam.app/help/).

The adapter is covered with mocked captures and an automatically generated video file. No physical camera or DroidCam configuration has been verified in this environment.

## Run and verify

```bash
python -m app.main
pytest
```

The camera adapter and preview utility are available in Phase 1. Detection, tracking, boundary events, face processing, persistence, notifications and the dashboard arrive in later phases. See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/INTERFACES.md](docs/INTERFACES.md).

## Local data

`data/` and `models/` are excluded from Git. Event image retention, enrolment/removal, access controls and real-deployment policies must be defined before real use; see [docs/DECISIONS.md](docs/DECISIONS.md).
