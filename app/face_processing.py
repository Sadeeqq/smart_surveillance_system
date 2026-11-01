"""Replaceable face detection, event capture quality checks and optional recognition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from app.boundary import BoundaryCrossingEvent
from app.tracker import TrackedPerson


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """Face box in source-frame pixels; landmarks follow YuNet's five-point order."""

    box_xyxy: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()
    clipped: bool = False


class FaceDetector(Protocol):
    def detect(self, frame: Any) -> list[FaceDetection]: ...


class FaceProcessingError(RuntimeError):
    """Raised for face-model setup or inference failures."""


def _create_yunet(model_path: str, score_threshold: float, nms_threshold: float,
                  top_k: int) -> Any:
    if not hasattr(cv2, "FaceDetectorYN"):
        raise FaceProcessingError("This OpenCV build does not provide FaceDetectorYN (YuNet)")
    if not Path(model_path).is_file():
        raise FaceProcessingError(
            f"YuNet model not found at {model_path!r}. Download the model using the Phase 5 README instructions."
        )
    try:
        return cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_threshold, top_k
        )
    except Exception as exc:
        raise FaceProcessingError(f"Could not load YuNet model {model_path!r}: {exc}") from exc


class YuNetFaceDetector:
    """OpenCV Zoo YuNet adapter. The model is kept outside Git and loaded on demand."""

    def __init__(self, model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
                 *, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 top_k: int = 5000, detector: Any | None = None,
                 detector_factory: Callable[..., Any] = _create_yunet) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.model_path = str(model_path)
        self._detector = detector if detector is not None else detector_factory(
            self.model_path, score_threshold, nms_threshold, top_k
        )

    def detect(self, frame: Any) -> list[FaceDetection]:
        if frame is None or getattr(frame, "size", 1) == 0:
            raise ValueError("frame must be a non-empty image")
        height, width = frame.shape[:2]
        try:
            self._detector.setInputSize((width, height))
            _ok, rows = self._detector.detect(frame)
        except Exception as exc:
            raise FaceProcessingError(f"YuNet inference failed: {exc}") from exc
        if rows is None:
            return []
        detections = []
        for row in np.asarray(rows):
            if len(row) < 15:
                continue
            x, y, box_width, box_height = (float(v) for v in row[:4])
            clipped = x < 0 or y < 0 or x + box_width > width or y + box_height > height
            left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
            right = min(width, int(np.ceil(x + box_width)))
            bottom = min(height, int(np.ceil(y + box_height)))
            landmarks = tuple((float(row[i]), float(row[i + 1])) for i in range(4, 14, 2))
            detections.append(FaceDetection((left, top, right, bottom), float(row[14]), landmarks,
                                            clipped=clipped))
        return detections


class CaptureStatus(str, Enum):
    CAPTURED = "captured"
    NO_FACE = "no_face"
    NOT_ASSOCIATED = "face_not_in_crossing_track"
    LOW_QUALITY = "low_quality"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FaceQuality:
    width: int
    height: int
    brightness: float
    blur_score: float
    face_confidence: float
    checks: Mapping[str, bool]

    @property
    def acceptable(self) -> bool:
        return all(self.checks.values())


@dataclass(frozen=True, slots=True)
class FaceCaptureResult:
    event_id: str
    track_id: int
    status: CaptureStatus
    reason: str
    detection: FaceDetection | None = None
    original_crop: Any | None = None
    quality: FaceQuality | None = None


class FaceCaptureService:
    """Pick the best face within the person box for the crossing track and quality-gate it."""

    def __init__(self, detector: FaceDetector, *, min_width: int = 64, min_height: int = 64,
                 min_brightness: float = 35.0, max_brightness: float = 220.0,
                 min_blur_score: float = 25.0, min_face_confidence: float = 0.6) -> None:
        if min_width < 1 or min_height < 1:
            raise ValueError("minimum crop dimensions must be positive")
        if not 0 <= min_brightness < max_brightness <= 255:
            raise ValueError("brightness limits must satisfy 0 <= min < max <= 255")
        if min_blur_score < 0 or not 0 <= min_face_confidence <= 1:
            raise ValueError("invalid blur or face confidence threshold")
        self.detector = detector
        self.min_width, self.min_height = min_width, min_height
        self.min_brightness, self.max_brightness = min_brightness, max_brightness
        self.min_blur_score, self.min_face_confidence = min_blur_score, min_face_confidence

    def capture(self, event: BoundaryCrossingEvent, frame: Any,
                tracked_people: Sequence[TrackedPerson]) -> FaceCaptureResult:
        base = dict(event_id=event.event_id, track_id=event.track_id)
        person = next((p for p in tracked_people if p.track_id == event.track_id), None)
        if person is None:
            return FaceCaptureResult(**base, status=CaptureStatus.UNAVAILABLE,
                                     reason="crossing_track_not_present_in_frame")
        if frame is None or getattr(frame, "size", 1) == 0:
            return FaceCaptureResult(**base, status=CaptureStatus.UNAVAILABLE, reason="frame_unavailable")
        try:
            faces = self.detector.detect(frame)
        except Exception as exc:
            return FaceCaptureResult(**base, status=CaptureStatus.UNAVAILABLE,
                                     reason=f"face_detector_unavailable: {exc}")
        px1, py1, px2, py2 = person.box_xyxy
        associated = [face for face in faces if
                      px1 <= (face.box_xyxy[0] + face.box_xyxy[2]) / 2 <= px2 and
                      py1 <= (face.box_xyxy[1] + face.box_xyxy[3]) / 2 <= py2]
        if not faces:
            return FaceCaptureResult(**base, status=CaptureStatus.NO_FACE, reason="no_face_detected")
        if not associated:
            return FaceCaptureResult(**base, status=CaptureStatus.NOT_ASSOCIATED,
                                     reason="no_detected_face_inside_crossing_track_box")
        face = max(associated, key=lambda item: item.confidence)
        x1, y1, x2, y2 = face.box_xyxy
        crop = frame[y1:y2, x1:x2].copy()
        quality = self._quality(crop, face)
        if not quality.acceptable:
            failures = ",".join(name for name, passed in quality.checks.items() if not passed)
            return FaceCaptureResult(**base, status=CaptureStatus.LOW_QUALITY,
                                     reason=f"quality_checks_failed:{failures}", detection=face,
                                     original_crop=crop, quality=quality)
        return FaceCaptureResult(**base, status=CaptureStatus.CAPTURED, reason="quality_checks_passed",
                                 detection=face, original_crop=crop, quality=quality)

    def _quality(self, crop: Any, face: FaceDetection) -> FaceQuality:
        height, width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        brightness = float(np.mean(gray)) if gray.size else 0.0
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
        checks = {
            "minimum_dimensions": width >= self.min_width and height >= self.min_height,
            "brightness": self.min_brightness <= brightness <= self.max_brightness,
            "sharpness": blur_score >= self.min_blur_score,
            "detector_confidence": face.confidence >= self.min_face_confidence,
            "face_visibility": not face.clipped and all(
                face.box_xyxy[0] <= x < face.box_xyxy[2] and
                face.box_xyxy[1] <= y < face.box_xyxy[3]
                for x, y in face.landmarks
            ) if face.landmarks else not face.clipped,
        }
        return FaceQuality(width, height, brightness, blur_score, face.confidence, checks)


@dataclass(frozen=True, slots=True)
class ImageVariants:
    original: Any
    clahe: Any


class FaceImageProcessor:
    """Optional luminance-only CLAHE; the source crop is never mutated or discarded."""

    def __init__(self, *, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        if clip_limit <= 0 or min(tile_grid_size) < 1:
            raise ValueError("CLAHE clip limit and tile dimensions must be positive")
        self.clip_limit, self.tile_grid_size = clip_limit, tile_grid_size

    def variants(self, original_crop: Any) -> ImageVariants:
        if original_crop is None or getattr(original_crop, "size", 1) == 0:
            raise ValueError("face crop must be non-empty")
        lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=self.tile_grid_size).apply(lightness)
        clahe = cv2.cvtColor(cv2.merge((enhanced, a, b)), cv2.COLOR_LAB2BGR)
        return ImageVariants(original_crop.copy(), clahe)

    @staticmethod
    def metrics(image: Any) -> dict[str, float]:
        """Return simple brightness/sharpness indicators for side-by-side evaluation."""
        if image is None or getattr(image, "size", 1) == 0:
            raise ValueError("face image must be non-empty")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return {
            "mean_brightness": float(np.mean(gray)),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }


class RecognitionStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    status: RecognitionStatus
    reason: str
    identity: str | None = None
    distance: float | None = None
    threshold: float | None = None
    model_name: str | None = None


class FaceRecognizer(Protocol):
    def recognize(self, crop: Any, enrolled_faces: Mapping[str, str]) -> RecognitionResult: ...


class DeepFaceFaceNetRecognizer:
    """Optional DeepFace/FaceNet comparison against explicitly enrolled reference images.

    Import and model initialization occur only on a recognition request. No face is enrolled
    automatically, and caller-provided references should be consented, local files.
    """

    def __init__(self, *, model_name: str = "Facenet", distance_margin: float = 0.05,
                 verifier: Callable[..., Mapping[str, Any]] | None = None) -> None:
        if not model_name.strip() or distance_margin < 0:
            raise ValueError("model name must be set and distance margin cannot be negative")
        self.model_name, self.distance_margin, self._verifier = model_name, distance_margin, verifier

    def recognize(self, crop: Any, enrolled_faces: Mapping[str, str]) -> RecognitionResult:
        if crop is None or getattr(crop, "size", 1) == 0:
            return RecognitionResult(RecognitionStatus.INCONCLUSIVE, "empty_face_crop", model_name=self.model_name)
        if not enrolled_faces:
            return RecognitionResult(RecognitionStatus.UNKNOWN, "no_enrolled_references", model_name=self.model_name)
        try:
            verifier = self._verifier
            if verifier is None:
                from deepface import DeepFace
                verifier = DeepFace.verify
        except ImportError:
            return RecognitionResult(RecognitionStatus.UNAVAILABLE,
                                     "optional_deepface_runtime_not_installed", model_name=self.model_name)

        matches: list[tuple[float, float, str]] = []
        try:
            for identity, reference in enrolled_faces.items():
                result = verifier(img1_path=crop, img2_path=str(reference), model_name=self.model_name,
                                  detector_backend="skip", enforce_detection=False, silent=True)
                distance, threshold = float(result["distance"]), float(result["threshold"])
                matches.append((distance, threshold, str(identity)))
        except Exception as exc:
            return RecognitionResult(RecognitionStatus.UNAVAILABLE,
                                     f"recognition_runtime_error:{exc}", model_name=self.model_name)
        matches.sort(key=lambda item: item[0])
        distance, threshold, identity = matches[0]
        if distance < threshold - self.distance_margin:
            return RecognitionResult(RecognitionStatus.KNOWN, "best_match_below_conservative_threshold",
                                     identity, distance, threshold, self.model_name)
        if distance > threshold + self.distance_margin:
            return RecognitionResult(RecognitionStatus.UNKNOWN, "no_reference_within_threshold",
                                     distance=distance, threshold=threshold, model_name=self.model_name)
        return RecognitionResult(RecognitionStatus.INCONCLUSIVE, "distance_near_threshold",
                                 distance=distance, threshold=threshold, model_name=self.model_name)
