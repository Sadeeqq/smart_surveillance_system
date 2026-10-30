from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import cv2
import pytest

from app.boundary import (
    BoundaryConfig,
    BoundaryDirection,
    BoundaryEngine,
    BoundaryStore,
    camera_identifier,
    signed_distance_to_boundary,
)
from app.boundary_test import _BoundaryMouseEditor, _default_boundary


def person(track_id=1, x=50, y=40, confidence=0.9):
    return SimpleNamespace(
        track_id=track_id,
        bottom_center=(x, y),
        confidence=confidence,
    )


def make_boundary(mode=BoundaryDirection.BOTH, enabled=True):
    # Horizontal line from A (left) to B (right); upper side is negative.
    return BoundaryConfig(
        camera_id="cam-1",
        boundary_id="default",
        start=(0.1, 0.5),
        end=(0.9, 0.5),
        direction_mode=mode,
        enabled=enabled,
    )


def prime_upper_side(engine, boundary, p=None, timestamp=None):
    p = p or person()
    engine.update(boundary, (100, 100), [p], timestamp=timestamp)
    when = timestamp + timedelta(seconds=1) if timestamp else None
    engine.update(boundary, (100, 100), [p], timestamp=when)


def cross_to_lower_side(engine, boundary, p=None, timestamp=None):
    p = p or person(y=60)
    engine.update(boundary, (100, 100), [p], timestamp=timestamp)
    when = timestamp + timedelta(seconds=1) if timestamp else None
    return engine.update(boundary, (100, 100), [p], timestamp=when)


def test_signed_distance_maps_normalized_line_to_frame_size():
    boundary = make_boundary()

    small = signed_distance_to_boundary((50, 60), boundary, (100, 100))
    large = signed_distance_to_boundary((100, 120), boundary, (200, 200))

    assert small == pytest.approx(10)
    assert large == pytest.approx(20)
    assert signed_distance_to_boundary((50, 40), boundary, (100, 100)) < 0


def test_invalid_boundaries_are_rejected():
    with pytest.raises(ValueError, match="different"):
        BoundaryConfig("cam", "line", (0.5, 0.5), (0.5, 0.5))
    with pytest.raises(ValueError, match="normalized"):
        BoundaryConfig("cam", "line", (-0.1, 0.5), (0.9, 0.5))
    with pytest.raises(ValueError, match="positive"):
        signed_distance_to_boundary((0, 0), make_boundary(), (0, 100))


def test_initial_appearance_on_one_side_is_not_a_crossing():
    engine = BoundaryEngine()
    boundary = make_boundary()

    for _ in range(5):
        assert engine.update(boundary, (100, 100), [person(y=40)]) == []


def test_single_frame_detection_on_other_side_is_debounced():
    engine = BoundaryEngine()
    boundary = make_boundary()
    prime_upper_side(engine, boundary)

    assert engine.update(boundary, (100, 100), [person(y=60)]) == []
    assert engine.update(boundary, (100, 100), [person(y=40)]) == []


def test_confirmed_crossing_emits_one_structured_event_and_rearms_on_return():
    engine = BoundaryEngine(cooldown_seconds=0)
    boundary = make_boundary()
    prime_upper_side(engine, boundary)

    events = cross_to_lower_side(engine, boundary)
    assert len(events) == 1
    event = events[0]
    assert event.camera_id == "cam-1"
    assert event.boundary_id == "default"
    assert event.track_id == 1
    assert event.direction is BoundaryDirection.A_TO_B
    assert event.detection_confidence == pytest.approx(0.9)
    assert event.to_dict()["direction"] == "a_to_b"
    assert engine.update(boundary, (100, 100), [person(y=60)]) == []

    engine.update(boundary, (100, 100), [person(y=40)])
    reverse = engine.update(boundary, (100, 100), [person(y=40)])
    assert len(reverse) == 1
    assert reverse[0].direction is BoundaryDirection.B_TO_A


def test_direction_mode_suppresses_disabled_direction():
    engine = BoundaryEngine(cooldown_seconds=0)
    boundary = make_boundary(mode=BoundaryDirection.B_TO_A)
    prime_upper_side(engine, boundary)

    assert cross_to_lower_side(engine, boundary) == []


def test_disabled_boundary_does_not_emit_events():
    engine = BoundaryEngine(cooldown_seconds=0)
    boundary = make_boundary(enabled=False)
    prime_upper_side(engine, boundary)

    assert cross_to_lower_side(engine, boundary) == []


def test_neutral_zone_jitter_does_not_emit_or_change_stable_side():
    engine = BoundaryEngine(cooldown_seconds=0)
    boundary = make_boundary()
    prime_upper_side(engine, boundary)

    for y in (50, 51, 49, 50, 40):
        assert engine.update(boundary, (100, 100), [person(y=y)]) == []

    assert len(cross_to_lower_side(engine, boundary)) == 1


def test_crossing_cooldown_suppresses_rapid_back_and_forth():
    engine = BoundaryEngine(cooldown_seconds=5)
    boundary = make_boundary()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prime_upper_side(engine, boundary, timestamp=base - timedelta(seconds=2))

    events = cross_to_lower_side(engine, boundary, timestamp=base)
    assert len(events) == 1
    engine.update(boundary, (100, 100), [person(y=40)], timestamp=base + timedelta(seconds=2))
    assert engine.update(boundary, (100, 100), [person(y=40)], timestamp=base + timedelta(seconds=3)) == []


def test_boundary_settings_are_saved_reloaded_per_camera_and_do_not_store_source(tmp_path):
    path = tmp_path / "boundaries.json"
    store = BoundaryStore(path)
    secret_source = "http://user:password@example.test/live"
    camera_id = camera_identifier(secret_source)
    boundary = BoundaryConfig(
        camera_id, "default", (0.2, 0.3), (0.8, 0.7), BoundaryDirection.A_TO_B
    )
    store.save(boundary)

    assert store.load(camera_id) == boundary
    assert store.load("another-camera") is None
    assert "password" not in path.read_text(encoding="utf-8")


def test_settings_store_reports_invalid_json(tmp_path):
    path = tmp_path / "boundaries.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read"):
        BoundaryStore(path).load("cam")


def test_engine_validates_debounce_settings():
    with pytest.raises(ValueError, match="neutral_zone_fraction"):
        BoundaryEngine(neutral_zone_fraction=0.5)
    with pytest.raises(ValueError, match="cooldown_seconds"):
        BoundaryEngine(cooldown_seconds=-1)


def test_mouse_editor_draws_and_drags_normalized_line():
    editor = _BoundaryMouseEditor(_default_boundary("cam"), 100, 100)

    editor.callback(cv2.EVENT_LBUTTONDOWN, 20, 5, 0, None)
    editor.callback(cv2.EVENT_LBUTTONUP, 20, 5, 0, None)
    editor.callback(cv2.EVENT_LBUTTONDOWN, 80, 95, 0, None)
    editor.callback(cv2.EVENT_LBUTTONUP, 80, 95, 0, None)

    assert editor.boundary.start == pytest.approx((0.2, 0.05))
    assert editor.boundary.end == pytest.approx((0.8, 0.95))

    editor.callback(cv2.EVENT_LBUTTONDOWN, 20, 5, 0, None)
    editor.callback(cv2.EVENT_MOUSEMOVE, 30, 10, 0, None)
    editor.callback(cv2.EVENT_LBUTTONUP, 30, 10, 0, None)

    assert editor.boundary.start == pytest.approx((0.3, 0.1))
