"""ByteTrack wrapper tests using a small Ultralytics-compatible fake model."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.tracker import (
    ByteTrackPersonTracker,
    TrackerError,
    TrackedPerson,
    TrajectoryAnnotator,
)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)


class FakeModel:
    names = {0: "person", 2: "car"}

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.options = None

    def track(self, **options):
        self.options = options
        if self.error:
            raise self.error
        return self.results


def test_tracker_returns_people_and_preserves_temporary_ids_and_anchor():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    model = FakeModel([SimpleNamespace(
        names={0: "Person", 2: "car"},
        boxes=FakeBoxes(
            xyxy=[[1, 2, 20, 35], [3, 4, 19, 30]],
            conf=[0.91, 0.99],
            cls=[0, 2],
            ids=[7, 8],
        ),
    )])
    tracker = ByteTrackPersonTracker(
        "weights.pt", confidence_threshold=0.1, image_size=320, device="cpu",
        tracker_config="bytetrack.yaml", model=model,
    )

    tracked = tracker.update(frame)

    assert tracked == [TrackedPerson(7, (1.0, 2.0, 20.0, 35.0), 0.91, 0, "Person")]
    assert tracked[0].bottom_center == (10.5, 35.0)
    assert model.options["persist"] is True
    assert model.options["tracker"] == "bytetrack.yaml"
    assert model.options["conf"] == 0.1
    assert model.options["classes"] == [0]
    assert model.options["imgsz"] == 320
    assert model.options["device"] == "cpu"


def test_untracked_boxes_and_missing_track_ids_return_empty_list():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    model = FakeModel([SimpleNamespace(
        names={0: "person"},
        boxes=FakeBoxes([[1, 2, 10, 12]], [0.8], [0], None),
    )])

    assert ByteTrackPersonTracker(model=model).update(frame) == []


def test_tracker_wraps_inference_errors_and_rejects_invalid_inputs():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    tracker = ByteTrackPersonTracker(model=FakeModel(error=RuntimeError("bad model")))

    with pytest.raises(TrackerError, match="ByteTrack update failed: bad model"):
        tracker.update(frame)
    with pytest.raises(ValueError, match="non-empty"):
        ByteTrackPersonTracker(model=FakeModel()).update(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="between 0 and 1"):
        ByteTrackPersonTracker(confidence_threshold=1.1, model=FakeModel())
    with pytest.raises(ValueError, match="at least 32"):
        ByteTrackPersonTracker(image_size=16, model=FakeModel())


def test_tracker_model_load_errors_are_actionable():
    def fail_to_load(_model_name):
        raise RuntimeError("weights unavailable")

    with pytest.raises(TrackerError, match="Could not load YOLO tracker model.*weights unavailable"):
        ByteTrackPersonTracker(model_factory=fail_to_load)


def test_trajectory_overlay_draws_box_and_path_without_mutating_frame():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    original = frame.copy()
    person = TrackedPerson(4, (8, 8, 24, 48), 0.8, 0, "person")
    annotator = TrajectoryAnnotator(trail_length=3)

    first = annotator.annotate(frame, [person])
    second = annotator.annotate(frame, [TrackedPerson(4, (10, 8, 26, 48), 0.82, 0, "person")])

    assert np.array_equal(frame, original)
    assert np.any(first != frame)
    assert np.any(second != frame)


def test_invalid_trajectory_length_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        TrajectoryAnnotator(trail_length=0)
