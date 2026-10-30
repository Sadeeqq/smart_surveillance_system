# System Architecture

The prototype is local-first. Camera-specific code is behind an adapter; the processing and event logic remain independent of the Streamlit interface. A virtual boundary crossing creates the primary event. Face processing and Telegram are optional enrichments and must not prevent local event creation or persistence.

```mermaid
flowchart LR
    C[Camera source] --> A[Camera adapter]
    A --> F[OpenCV frames]
    F --> D[YOLO person detector]
    D --> T[ByteTrack tracker]
    T --> B[Normalized virtual boundary engine]
    B -->|direction-checked crossing candidate| E[Event manager]
    B --> X[Face detection and quality gate]
    X --> P[Optional CLAHE and recognition]
    P --> E
    E --> DB[(SQLite)]
    E --> FS[Local image files]
    E --> N[Optional Telegram]
    E --> UI[Streamlit dashboard]
```

The dashboard calls services and displays their status; it does not own detection, tracking, geometry, or persistence logic. Tracker IDs are temporary. A missing or unknown face is not a threat classification. The initial DroidCam-over-Wi-Fi path is replaceable by a webcam, test video, or another adapter that meets the camera contract. Actual operating-system and camera combinations are supported only after being tested and documented.

Phase 4 keeps geometry and crossing state in `app/boundary.py`, independent of the OpenCV editor. The editor is a temporary operator tool; Phase 7 can replace it with Streamlit controls while reusing the same boundary configuration and engine. Its JSON settings file is local-only until the Phase 6 database settings layer is implemented.

See [INTERFACES.md](INTERFACES.md) for the initial contracts and [DECISIONS.md](DECISIONS.md) for unresolved choices.
