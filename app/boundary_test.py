"""Live boundary editor and crossing preview (press q or Esc to quit)."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import cv2

from app.boundary import (
    BoundaryConfig,
    BoundaryDirection,
    BoundaryEngine,
    BoundaryStore,
    camera_identifier,
)
from app.camera import CameraState, OpenCVCameraSource
from app.config import Settings
from app.face_processing import (
    FaceCaptureService,
    FaceImageProcessor,
    FaceProcessingError,
    DeepFaceFaceNetRecognizer,
    RecognitionResult,
    RecognitionStatus,
    YuNetFaceDetector,
)
from app.tracker import ByteTrackPersonTracker, TrackerError, TrajectoryAnnotator


class _BoundaryMouseEditor:
    """Mouse controls for creating a line and dragging an endpoint or midpoint."""

    def __init__(self, boundary: BoundaryConfig, width: int, height: int) -> None:
        self.boundary = boundary
        self.width = width
        self.height = height
        self.drawing = False
        self.first_point: tuple[float, float] | None = None
        self.drag_target: str | None = None
        self.last_position: tuple[float, float] | None = None

    def callback(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        point = (min(1.0, max(0.0, x / self.width)), min(1.0, max(0.0, y / self.height)))
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                if self.first_point is None:
                    self.first_point = point
                else:
                    start = self.first_point
                    if start == point:
                        return
                    self._replace(start=start, end=point)
                    self.first_point = None
                    self.drawing = False
                return
            self.drag_target = self._nearest_target(x, y)
            self.last_position = point
            if self.drag_target is None:
                self.drawing = True
                self.first_point = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_target:
            if self.drag_target == "start":
                self._replace(start=point)
            elif self.drag_target == "end":
                self._replace(end=point)
            elif self.drag_target == "line" and self.last_position is not None:
                dx, dy = point[0] - self.last_position[0], point[1] - self.last_position[1]
                start = (min(1.0, max(0.0, self.boundary.start[0] + dx)),
                         min(1.0, max(0.0, self.boundary.start[1] + dy)))
                end = (min(1.0, max(0.0, self.boundary.end[0] + dx)),
                       min(1.0, max(0.0, self.boundary.end[1] + dy)))
                self._replace(start=start, end=end)
            self.last_position = point
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_target = None
            self.last_position = None

    def _replace(self, **changes: object) -> None:
        try:
            self.boundary = replace(self.boundary, **changes)
        except ValueError:
            # Ignore a mouse position that would collapse the line endpoints.
            pass

    def _nearest_target(self, x: int, y: int) -> str | None:
        points = {
            "start": (self.boundary.start[0] * self.width, self.boundary.start[1] * self.height),
            "end": (self.boundary.end[0] * self.width, self.boundary.end[1] * self.height),
            "line": ((self.boundary.start[0] + self.boundary.end[0]) * self.width / 2,
                     (self.boundary.start[1] + self.boundary.end[1]) * self.height / 2),
        }
        target, distance = min(
            ((name, (x - px) ** 2 + (y - py) ** 2) for name, (px, py) in points.items()),
            key=lambda item: item[1],
        )
        return target if distance <= 35**2 else None


def _default_boundary(camera_id: str) -> BoundaryConfig:
    return BoundaryConfig(
        camera_id=camera_id,
        boundary_id="default",
        start=(0.1, 0.5),
        end=(0.9, 0.5),
        direction_mode=BoundaryDirection.BOTH,
        enabled=False,
    )


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Camera index, V4L2 path, stream URL or video path")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--confidence", type=float, default=settings.tracking_confidence_threshold)
    parser.add_argument("--image-size", type=int, default=settings.detection_image_size)
    parser.add_argument("--device", default=settings.detection_device)
    parser.add_argument("--tracker", default=settings.tracker_config)
    parser.add_argument("--trail-length", type=int, default=settings.tracking_trail_length)
    parser.add_argument("--boundary-file", default="data/boundaries.json",
                        help="Local JSON file used to save per-camera line settings")
    parser.add_argument("--face-capture", action="store_true",
                        help="Try a consented, in-memory YuNet face capture on each crossing")
    parser.add_argument("--face-model", default="models/face_detection_yunet_2023mar.onnx",
                        help="Local YuNet ONNX model path (model weights are not committed)")
    parser.add_argument("--recognize", action="store_true",
                        help="Optionally compare usable face crops with explicit local references")
    parser.add_argument("--enrolled-face", action="append", default=[], metavar="NAME=PATH",
                        help="Consented reference image; repeat for multiple references (requires --recognize)")
    args = parser.parse_args()
    if args.recognize and not args.face_capture:
        parser.error("--recognize requires --face-capture")
    if args.enrolled_face and not args.recognize:
        parser.error("--enrolled-face requires --recognize")
    enrolled_faces: dict[str, str] = {}
    for entry in args.enrolled_face:
        name, separator, path = entry.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("--enrolled-face must use NAME=PATH")
        if name in enrolled_faces:
            parser.error(f"duplicate enrolled reference name: {name}")
        enrolled_faces[name] = path

    camera_id = camera_identifier(args.source)
    store = BoundaryStore(args.boundary_file)
    try:
        boundary = store.load(camera_id) or _default_boundary(camera_id)
    except ValueError as exc:
        parser.error(str(exc))

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
        engine = BoundaryEngine(cooldown_seconds=settings.event_cooldown_seconds)
        face_capture = (FaceCaptureService(YuNetFaceDetector(args.face_model))
                        if args.face_capture else None)
        face_recognizer = DeepFaceFaceNetRecognizer() if args.recognize else None
    except (TrackerError, FaceProcessingError, ValueError) as exc:
        camera.release()
        print(exc)
        return 1

    started = time.monotonic()
    frame_count = 0
    editor: _BoundaryMouseEditor | None = None
    window = "Smart Surveillance Boundary Editor"
    cv2.namedWindow(window)
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                if camera.status().state is CameraState.ERROR:
                    print(camera.status().message)
                    return 1
                continue
            height, width = frame.shape[:2]
            if editor is None:
                editor = _BoundaryMouseEditor(boundary, width, height)
                cv2.setMouseCallback(window, editor.callback)
            tracked_people = tracker.update(frame)
            previous_boundary = boundary
            boundary = editor.boundary
            if boundary != previous_boundary:
                # Relearn which side each current track occupies after a line
                # is edited, so moving the boundary itself cannot look like a crossing.
                engine.reset()
            events = engine.update(boundary, (width, height), tracked_people)
            frame_count += 1
            fps = frame_count / max(time.monotonic() - started, 1e-6)
            preview = annotator.annotate(frame, tracked_people)

            start_px = (round(boundary.start[0] * width), round(boundary.start[1] * height))
            end_px = (round(boundary.end[0] * width), round(boundary.end[1] * height))
            color = (50, 230, 50) if boundary.enabled else (0, 190, 255)
            cv2.line(preview, start_px, end_px, color, 3)
            cv2.circle(preview, start_px, 7, (255, 100, 20), -1)
            cv2.circle(preview, end_px, 7, (20, 100, 255), -1)
            if editor.first_point:
                pending = (round(editor.first_point[0] * width), round(editor.first_point[1] * height))
                cv2.circle(preview, pending, 7, (255, 100, 20), -1)
            state = "ENABLED" if boundary.enabled else "DISABLED"
            controls = f"{state} | direction: {boundary.direction_mode.value} | {fps:.1f} FPS"
            cv2.putText(preview, controls, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, color, 2, cv2.LINE_AA)
            cv2.putText(preview, "Drag endpoint to rotate/resize; drag line to move | click to draw",
                        (12, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(preview, "1 A-to-B | 2 B-to-A | 3 Both | e Enable | s Save | r Reset | q Quit",
                        (12, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, preview)
            for event in events:
                print(f"BOUNDARY CROSSING: {event.to_dict()}")
                if face_capture is not None:
                    result = face_capture.capture(event, frame, tracked_people)
                    quality = result.quality
                    variants = (FaceImageProcessor().variants(result.original_crop)
                                if result.original_crop is not None else None)
                    summary = {
                        "event_id": result.event_id,
                        "track_id": result.track_id,
                        "capture_status": result.status.value,
                        "reason": result.reason,
                        "face_box": result.detection.box_xyxy if result.detection else None,
                        "quality": ({"width": quality.width, "height": quality.height,
                                     "brightness": round(quality.brightness, 1),
                                     "laplacian_variance": round(quality.blur_score, 1),
                                     "checks": dict(quality.checks)} if quality else None),
                        "image_metrics_original": (FaceImageProcessor.metrics(variants.original)
                                                   if variants else None),
                        "image_metrics_clahe": (FaceImageProcessor.metrics(variants.clahe)
                                                if variants else None),
                    }
                    print(f"FACE CAPTURE: {summary}")
                    if face_recognizer is not None:
                        if result.status.value != "captured":
                            recognition = RecognitionResult(
                                RecognitionStatus.INCONCLUSIVE,
                                f"face_capture_{result.status.value}",
                                model_name=face_recognizer.model_name,
                            )
                        else:
                            recognition = face_recognizer.recognize(
                                result.original_crop, enrolled_faces
                            )
                        print("FACE RECOGNITION: " + str({
                            "status": recognition.status.value,
                            "reason": recognition.reason,
                            "identity": recognition.identity,
                            "distance": recognition.distance,
                            "threshold": recognition.threshold,
                            "model": recognition.model_name,
                        }))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                editor.boundary = replace(boundary, direction_mode=BoundaryDirection.A_TO_B)
            elif key == ord("2"):
                editor.boundary = replace(boundary, direction_mode=BoundaryDirection.B_TO_A)
            elif key == ord("3"):
                editor.boundary = replace(boundary, direction_mode=BoundaryDirection.BOTH)
            elif key == ord("e"):
                editor.boundary = replace(boundary, enabled=not boundary.enabled)
            elif key == ord("s"):
                store.save(boundary)
                print(f"Saved boundary settings to {store.path} for camera {camera_id}")
            elif key == ord("r"):
                editor.boundary = _default_boundary(camera_id)
                engine.reset()
                store.save(editor.boundary)
                print("Reset and saved the disabled default boundary")
            elif key in (ord("q"), 27):
                return 0
    except (TrackerError, ValueError, OSError) as exc:
        print(exc)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
