"""Temporary person tracking backed by Ultralytics' ByteTrack integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import cv2

from app.detector import _load_yolo


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    """A person box associated across frames by a temporary tracker ID."""

    track_id: int
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Return the bottom-centre anchor used by later boundary logic."""
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, y2)


class TrackerError(RuntimeError):
    """Raised when the optional tracking runtime cannot load or update."""


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, Mapping):
        value = names.get(class_id)
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        value = names[class_id] if 0 <= class_id < len(names) else None
    else:
        value = None
    return str(value) if value is not None else None


class ByteTrackPersonTracker:
    """Track only people across consecutive frames from one video source.

    Ultralytics runs its detector and ByteTrack in one supported ``track`` call.
    Create one tracker per stream; the returned IDs are temporary and must not
    be interpreted as a person's real-world identity.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        confidence_threshold: float = 0.1,
        image_size: int = 640,
        device: str = "cpu",
        tracker_config: str = "bytetrack.yaml",
        model: Any | None = None,
        model_factory: Callable[[str], Any] = _load_yolo,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        if not device.strip():
            raise ValueError("device cannot be empty")
        if not tracker_config.strip():
            raise ValueError("tracker_config cannot be empty")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.tracker_config = tracker_config
        try:
            self._model = model if model is not None else model_factory(model_name)
        except Exception as exc:
            raise TrackerError(f"Could not load YOLO tracker model {model_name!r}: {exc}") from exc

    def update(self, frame: Any) -> list[TrackedPerson]:
        """Update tracking with the next frame and return confirmed person IDs."""
        if frame is None or getattr(frame, "size", 1) == 0:
            raise ValueError("frame must be a non-empty image")
        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence_threshold,
                classes=[0],
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise TrackerError(f"ByteTrack update failed: {exc}") from exc

        tracked: list[TrackedPerson] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "id", None) is None:
                continue
            names = getattr(result, "names", None) or getattr(self._model, "names", None)
            coordinates = _values(boxes.xyxy)
            confidences = _values(boxes.conf)
            class_ids = _values(boxes.cls)
            track_ids = _values(boxes.id)
            for raw_box, raw_conf, raw_class_id, raw_track_id in zip(
                coordinates, confidences, class_ids, track_ids
            ):
                class_id = int(raw_class_id)
                class_name = _class_name(names, class_id)
                if class_name is None or class_name.casefold() != "person" or len(raw_box) != 4:
                    continue
                tracked.append(
                    TrackedPerson(
                        track_id=int(raw_track_id),
                        box_xyxy=tuple(float(value) for value in raw_box),
                        confidence=float(raw_conf),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
        return tracked


class TrajectoryAnnotator:
    """Draw track labels and bounded bottom-centre trajectories on a copy."""

    def __init__(self, trail_length: int = 30) -> None:
        if trail_length < 1:
            raise ValueError("trail_length must be at least 1")
        self.trail_length = trail_length
        self._frame_index = 0
        self._points: dict[int, tuple[int, deque[tuple[int, int]]]] = {}

    def annotate(self, frame: Any, tracked_people: Sequence[TrackedPerson]) -> Any:
        if frame is None or getattr(frame, "size", 1) == 0:
            raise ValueError("frame must be a non-empty image")
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        self._frame_index += 1
        active_ids = set()
        for person in tracked_people:
            track_id = person.track_id
            active_ids.add(track_id)
            x1, y1, x2, y2 = person.box_xyxy
            left = max(0, min(width - 1, round(x1)))
            top = max(0, min(height - 1, round(y1)))
            right = max(0, min(width - 1, round(x2)))
            bottom = max(0, min(height - 1, round(y2)))
            anchor_x, anchor_y = person.bottom_center
            point = (max(0, min(width - 1, round(anchor_x))),
                     max(0, min(height - 1, round(anchor_y))))
            last_seen, points = self._points.get(
                track_id, (self._frame_index, deque(maxlen=self.trail_length))
            )
            if self._frame_index - last_seen > self.trail_length:
                points.clear()
            points.append(point)
            self._points[track_id] = (self._frame_index, points)

            cv2.rectangle(annotated, (left, top), (right, bottom), (60, 235, 60), 2)
            cv2.putText(
                annotated,
                f"track {track_id} | {person.confidence:.2f}",
                (left, max(16, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 235, 60),
                2,
                cv2.LINE_AA,
            )
            trail = list(points)
            for previous, current in zip(trail, trail[1:]):
                cv2.line(annotated, previous, current, (255, 180, 0), 2)

        for track_id, (last_seen, _) in list(self._points.items()):
            if track_id not in active_ids and self._frame_index - last_seen > self.trail_length:
                del self._points[track_id]
        return annotated
