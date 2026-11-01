"""Preview person detection, ByteTrack IDs and face detection together, without boundaries."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.camera import CameraState, OpenCVCameraSource
from app.config import Settings
from app.face_processing import FaceProcessingError, YuNetFaceDetector
from app.tracker import ByteTrackPersonTracker, TrackerError, TrajectoryAnnotator


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Camera index, V4L2 path, stream URL or video file")
    parser.add_argument("--model", default=settings.model_name,
                        help="Ultralytics YOLO model used for person detection/tracking")
    parser.add_argument("--face-model", default="models/face_detection_yunet_2023mar.onnx",
                        help="Path to the local YuNet ONNX face model")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum YuNet face confidence (0 to 1)")
    parser.add_argument("--image-size", type=int, default=settings.detection_image_size,
                        help="YOLO inference image size")
    parser.add_argument("--confidence", type=float, default=settings.tracking_confidence_threshold,
                        help="YOLO person detection confidence threshold")
    parser.add_argument("--device", default=settings.detection_device,
                        help="YOLO inference device, usually cpu")
    parser.add_argument("--tracker", default=settings.tracker_config,
                        help="Ultralytics tracker configuration")
    args = parser.parse_args()
    if not Path(args.face_model).is_file():
        parser.error(f"YuNet model not found: {args.face_model}. See the Phase 5 README download command.")

    camera = OpenCVCameraSource(args.source, reconnect_attempts=settings.camera_reconnect_attempts)
    if not camera.connect():
        print(camera.status().message)
        return 1
    try:
        tracker = ByteTrackPersonTracker(
            args.model,
            confidence_threshold=args.confidence,
            image_size=args.image_size,
            device=args.device,
            tracker_config=args.tracker,
        )
        face_detector = YuNetFaceDetector(args.face_model, score_threshold=args.threshold)
        annotator = TrajectoryAnnotator(settings.tracking_trail_length)
    except (FaceProcessingError, TrackerError, ValueError) as exc:
        camera.release()
        print(exc)
        return 1

    window = "Smart Surveillance Person + Face Detection + Tracking"
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

            tracked_people = tracker.update(frame)
            faces = face_detector.detect(frame)
            preview = annotator.annotate(frame, tracked_people)
            for face in faces:
                x1, y1, x2, y2 = face.box_xyxy
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 160, 0), 2)
                cv2.putText(preview, f"face {face.confidence:.2f}", (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2, cv2.LINE_AA)
                for x, y in face.landmarks:
                    cv2.circle(preview, (round(x), round(y)), 2, (0, 180, 255), -1)

            frame_count += 1
            fps = frame_count / max(time.monotonic() - started, 1e-6)
            label = f"people: {len(tracked_people)} | faces: {len(faces)} | {fps:.1f} FPS"
            cv2.putText(preview, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 235, 60), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except (FaceProcessingError, TrackerError) as exc:
        print(exc)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
