"""Person detector tests with a small fake Ultralytics-compatible model."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.detector import Detection, DetectorError, YoloPersonDetector, draw_detections


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.options = None

    def predict(self, **options):
        self.options = options
        if self.error:
            raise self.error
        return self.results


def test_detect_returns_only_persons_meeting_confidence_threshold():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    model = FakeModel([SimpleNamespace(
        names={0: "Person", 2: "car"},
        boxes=FakeBoxes(
            xyxy=[[1, 2, 20, 35], [3, 4, 19, 30], [5, 5, 10, 10]],
            conf=[0.91, 0.99, 0.2],
            cls=[0, 2, 0],
        ),
    )])
    detector = YoloPersonDetector(
        "weights.pt", confidence_threshold=0.5, image_size=320, device="cpu", model=model
    )

    detections = detector.detect(frame)

    assert detections == [Detection((1.0, 2.0, 20.0, 35.0), 0.91, 0, "Person")]
    assert model.options["conf"] == 0.5
    assert model.options["imgsz"] == 320
    assert model.options["device"] == "cpu"
    assert model.options["verbose"] is False


def test_detector_loads_model_lazily_through_factory():
    model = FakeModel()
    loaded = []
    detector = YoloPersonDetector(model_factory=lambda name: loaded.append(name) or model)

    assert loaded == ["yolo11n.pt"]
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


def test_inference_errors_are_reported_as_detector_errors():
    detector = YoloPersonDetector(model=FakeModel(error=RuntimeError("bad weights")))

    with pytest.raises(DetectorError, match="YOLO inference failed: bad weights"):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_empty_frames_and_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        YoloPersonDetector(model=FakeModel()).detect(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="between 0 and 1"):
        YoloPersonDetector(confidence_threshold=1.1, model=FakeModel())
    with pytest.raises(ValueError, match="at least 32"):
        YoloPersonDetector(image_size=16, model=FakeModel())


def test_debug_overlay_marks_boxes_and_does_not_mutate_frame():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    before = frame.copy()
    detection = Detection((2, 2, 24, 28), 0.8, 0, "person")

    overlay = draw_detections(frame, [detection])

    assert overlay.shape == frame.shape
    assert np.array_equal(frame, before)
    assert np.any(overlay != frame)


def test_overlay_rejects_empty_frame():
    with pytest.raises(ValueError, match="non-empty"):
        draw_detections(np.empty((0, 0, 3), dtype=np.uint8), [])
