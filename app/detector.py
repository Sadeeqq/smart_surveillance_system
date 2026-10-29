"""Person detection and debug annotation backed by an optional Ultralytics YOLO model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import cv2


@dataclass(frozen=True, slots=True)
class Detection:
    """One frame-relative object detection, with coordinates in source pixels."""

    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class DetectorError(RuntimeError):
    """Raised when the optional YOLO runtime cannot load or infer."""


def _load_yolo(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "Person detection needs the optional Ultralytics runtime. "
            "Install it with: python -m pip install -r requirements-detection.txt"
        ) from exc
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise DetectorError(f"Could not load YOLO model {model_name!r}: {exc}") from exc


def _values(value: Any) -> list[Any]:
    """Convert a Torch tensor, NumPy array or simple sequence to Python values."""
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


class YoloPersonDetector:
    """Return person detections from an Ultralytics-compatible YOLO model.

    Model loading is delayed until this class is instantiated. Pass ``model`` or
    ``model_factory`` in tests to exercise filtering without downloading weights.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        device: str = "cpu",
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
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self._model = model if model is not None else model_factory(model_name)

    def detect(self, frame: Any) -> list[Detection]:
        """Run inference and return only detections labelled ``person``."""
        if frame is None or getattr(frame, "size", 1) == 0:
            raise ValueError("frame must be a non-empty image")
        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        detections: list[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or getattr(self._model, "names", None)
            coordinates = _values(boxes.xyxy)
            confidences = _values(boxes.conf)
            class_ids = _values(boxes.cls)
            for raw_box, raw_confidence, raw_class_id in zip(coordinates, confidences, class_ids):
                class_id = int(raw_class_id)
                class_name = _class_name(names, class_id)
                confidence = float(raw_confidence)
                if class_name is None or class_name.casefold() != "person":
                    continue
                if confidence < self.confidence_threshold or len(raw_box) != 4:
                    continue
                box = tuple(float(value) for value in raw_box)
                detections.append(
                    Detection(box_xyxy=box, confidence=confidence, class_id=class_id,
                              class_name=class_name)
                )
        return detections


def draw_detections(frame: Any, detections: Sequence[Detection]) -> Any:
    """Return a copy of ``frame`` annotated with person boxes and confidence."""
    if frame is None or getattr(frame, "size", 1) == 0:
        raise ValueError("frame must be a non-empty image")
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        left = max(0, min(width - 1, round(x1)))
        top = max(0, min(height - 1, round(y1)))
        right = max(0, min(width - 1, round(x2)))
        bottom = max(0, min(height - 1, round(y2)))
        color = (60, 235, 60)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"person {detection.confidence:.2f}"
        text_y = max(16, top - 6)
        cv2.putText(annotated, label, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    return annotated
