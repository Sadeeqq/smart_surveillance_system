"""Face-only preview using YuNet (press q or Esc to close)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.camera import CameraState, OpenCVCameraSource
from app.face_processing import FaceProcessingError, YuNetFaceDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Camera index, V4L2 path, stream URL or video file")
    parser.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx",
                        help="Path to the local YuNet ONNX model")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum YuNet face confidence (0 to 1)")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f"YuNet model not found: {args.model}. See the Phase 5 README download command.")

    camera = OpenCVCameraSource(args.source)
    if not camera.connect():
        print(camera.status().message)
        return 1
    try:
        detector = YuNetFaceDetector(args.model, score_threshold=args.threshold)
    except (FaceProcessingError, ValueError) as exc:
        camera.release()
        print(exc)
        return 1

    window = "Smart Surveillance Face Detection"
    started = time.monotonic()
    frame_count = 0
    cv2.namedWindow(window)
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                if camera.status().state is CameraState.ERROR:
                    print(camera.status().message)
                    return 1
                continue

            detections = detector.detect(frame)
            preview = frame.copy()
            for face in detections:
                x1, y1, x2, y2 = face.box_xyxy
                cv2.rectangle(preview, (x1, y1), (x2, y2), (60, 235, 60), 2)
                cv2.putText(preview, f"face {face.confidence:.2f}", (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 235, 60), 2, cv2.LINE_AA)
                for x, y in face.landmarks:
                    cv2.circle(preview, (round(x), round(y)), 2, (0, 180, 255), -1)

            frame_count += 1
            fps = frame_count / max(time.monotonic() - started, 1e-6)
            label = f"faces: {len(detections)} | {fps:.1f} FPS | threshold: {args.threshold:.2f}"
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 235, 60), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except FaceProcessingError as exc:
        print(exc)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
