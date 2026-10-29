"""Live or recorded-video ByteTrack preview (press q or Esc to quit)."""

from __future__ import annotations

import argparse
import time

import cv2

from app.camera import CameraState, OpenCVCameraSource
from app.config import Settings
from app.tracker import ByteTrackPersonTracker, TrackerError, TrajectoryAnnotator


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Camera index, V4L2 path, stream URL or video path")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--confidence", type=float, default=settings.tracking_confidence_threshold)
    parser.add_argument("--image-size", type=int, default=settings.detection_image_size)
    parser.add_argument("--device", default=settings.detection_device,
                        help="Ultralytics device, e.g. cpu or 0 (default: cpu)")
    parser.add_argument("--tracker", default=settings.tracker_config,
                        help="Ultralytics tracker config (default: bytetrack.yaml)")
    parser.add_argument("--trail-length", type=int, default=settings.tracking_trail_length)
    args = parser.parse_args()

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
        annotator = TrajectoryAnnotator(args.trail_length)
    except (TrackerError, ValueError) as exc:
        camera.release()
        print(exc)
        return 1

    started = time.monotonic()
    frame_count = 0
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                if camera.status().state is CameraState.ERROR:
                    print(camera.status().message)
                    return 1
                continue
            try:
                tracked_people = tracker.update(frame)
            except TrackerError as exc:
                print(exc)
                return 1
            frame_count += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            fps = frame_count / elapsed
            preview = annotator.annotate(frame, tracked_people)
            cv2.putText(preview, f"people: {len(tracked_people)} | {fps:.1f} FPS", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2, cv2.LINE_AA)
            cv2.imshow("Smart Surveillance ByteTrack Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
