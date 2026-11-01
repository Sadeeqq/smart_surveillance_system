from datetime import datetime, timezone

import numpy as np

from app.boundary import BoundaryCrossingEvent, BoundaryDirection
from app.face_processing import (
    CaptureStatus,
    DeepFaceFaceNetRecognizer,
    FaceCaptureService,
    FaceDetection,
    FaceImageProcessor,
    RecognitionStatus,
    YuNetFaceDetector,
)
from app.tracker import TrackedPerson


class FakeFaceDetector:
    def __init__(self, detections=None, error=None):
        self.detections = detections or []
        self.error = error

    def detect(self, _frame):
        if self.error:
            raise self.error
        return self.detections


def _event(track_id=8):
    return BoundaryCrossingEvent("event-1", datetime.now(timezone.utc), "cam", track_id,
                                  "door", BoundaryDirection.A_TO_B, 0.91)


def _person(track_id=8):
    return TrackedPerson(track_id, (10, 10, 100, 100), 0.91, 0, "person")


def _texture_frame():
    rng = np.random.default_rng(3)
    return rng.integers(40, 210, size=(120, 120, 3), dtype=np.uint8)


def test_capture_associates_face_to_event_track_and_keeps_original_crop():
    frame = _texture_frame()
    face = FaceDetection((15, 15, 95, 95), 0.95)
    result = FaceCaptureService(FakeFaceDetector([face])).capture(_event(), frame, [_person()])

    assert result.status is CaptureStatus.CAPTURED
    assert result.reason == "quality_checks_passed"
    assert result.original_crop.shape == (80, 80, 3)
    assert result.quality.acceptable
    assert result.detection == face


def test_capture_reports_no_face_track_mismatch_and_detector_failure_explicitly():
    frame = _texture_frame()
    no_face = FaceCaptureService(FakeFaceDetector()).capture(_event(), frame, [_person()])
    wrong_track = FaceCaptureService(FakeFaceDetector([FaceDetection((12, 12, 90, 90), .9)])).capture(
        _event(), frame, [_person(track_id=3)]
    )
    unavailable = FaceCaptureService(FakeFaceDetector(error=RuntimeError("offline"))).capture(
        _event(), frame, [_person()]
    )

    assert no_face.status is CaptureStatus.NO_FACE
    assert no_face.reason == "no_face_detected"
    assert wrong_track.status is CaptureStatus.UNAVAILABLE
    assert wrong_track.reason == "crossing_track_not_present_in_frame"
    assert unavailable.status is CaptureStatus.UNAVAILABLE
    assert "offline" in unavailable.reason


def test_face_outside_track_and_low_quality_are_not_promoted_to_capture():
    frame = _texture_frame()
    outside = FaceCaptureService(FakeFaceDetector([FaceDetection((101, 20, 119, 60), .9)])).capture(
        _event(), frame, [_person()]
    )
    small = FaceCaptureService(FakeFaceDetector([FaceDetection((15, 15, 45, 45), .95)])).capture(
        _event(), frame, [_person()]
    )

    assert outside.status is CaptureStatus.NOT_ASSOCIATED
    assert small.status is CaptureStatus.LOW_QUALITY
    assert "minimum_dimensions" in small.reason
    assert small.original_crop is not None


def test_yunet_adapter_maps_rows_and_sets_dynamic_input_dimensions():
    class FakeYuNet:
        def setInputSize(self, size):
            self.size = size

        def detect(self, _frame):
            # x, y, w, h, five landmarks, confidence
            return True, np.asarray([[2, 3, 10, 12, *([1] * 10), .88]], dtype=float)

    model = FakeYuNet()
    adapter = YuNetFaceDetector(detector=model)
    detections = adapter.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert model.size == (32, 24)
    assert detections == [FaceDetection((2, 3, 12, 15), .88, ((1, 1),) * 5)]


def test_clahe_returns_separate_original_and_luminance_enhanced_images():
    frame = np.full((80, 80, 3), 70, dtype=np.uint8)
    frame[20:60, 20:60] = 80
    original = frame.copy()

    variants = FaceImageProcessor().variants(frame)

    assert np.array_equal(frame, original)
    assert np.array_equal(variants.original, original)
    assert variants.clahe.shape == original.shape
    assert not np.shares_memory(variants.clahe, frame)


def test_optional_facenet_recognizer_preserves_unknown_and_inconclusive_outcomes():
    crop = np.ones((20, 20, 3), dtype=np.uint8)
    missing_runtime = DeepFaceFaceNetRecognizer().recognize(crop, {"consented": "face.jpg"})
    no_reference = DeepFaceFaceNetRecognizer(verifier=lambda **_: {}).recognize(crop, {})
    near_threshold = DeepFaceFaceNetRecognizer(
        verifier=lambda **_: {"distance": .60, "threshold": .60}
    ).recognize(crop, {"consented": "face.jpg"})
    known = DeepFaceFaceNetRecognizer(
        verifier=lambda **_: {"distance": .3, "threshold": .6}
    ).recognize(crop, {"consented": "face.jpg"})

    assert missing_runtime.status is RecognitionStatus.UNAVAILABLE
    assert no_reference.status is RecognitionStatus.UNKNOWN
    assert near_threshold.status is RecognitionStatus.INCONCLUSIVE
    assert known.status is RecognitionStatus.KNOWN
    assert known.identity == "consented"
