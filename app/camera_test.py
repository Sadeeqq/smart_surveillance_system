"""Small live camera preview and FPS diagnostic (press q or Esc to quit)."""

from __future__ import annotations

import argparse
import time

import cv2

from app.camera import CameraState, OpenCVCameraSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Camera index, virtual camera path, stream URL or video path")
    parser.add_argument("--reconnect-attempts", type=int, default=3)
    args = parser.parse_args()

    camera = OpenCVCameraSource(args.source, reconnect_attempts=args.reconnect_attempts)
    if not camera.connect():
        print(camera.status().message)
        return 1

    frames = 0
    started = time.monotonic()
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                if camera.status().state == CameraState.ERROR:
                    print(camera.status().message)
                    return 1
                continue
            frames += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            fps = frames / elapsed
            cv2.putText(frame, f"{camera.status().state.value} | {fps:.1f} FPS", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2, cv2.LINE_AA)
            cv2.imshow("Smart Surveillance Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
