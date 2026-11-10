from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.alerts import AlertResult, TelegramNotifier
from app.boundary import BoundaryConfig, BoundaryCrossingEvent, BoundaryDirection
from app.database import DatabaseError, EventRecord, SQLiteRepository
from app.event_manager import EventManager
from app.face_processing import (
    CaptureStatus,
    FaceCaptureResult,
    FaceDetection,
    FaceQuality,
    RecognitionResult,
    RecognitionStatus,
)
from app.image_storage import LocalImageStore


def _crossing(event_id="event-test"):
    return BoundaryCrossingEvent(event_id, datetime.now(timezone.utc), "camera-hash", 12,
                                 "entrance", BoundaryDirection.A_TO_B, 0.87)


def _record(event_id="event-test"):
    crossing = _crossing(event_id)
    return EventRecord(
        crossing.event_id, crossing.timestamp.isoformat(), crossing.camera_id, crossing.track_id,
        crossing.boundary_id, crossing.direction.value, crossing.detection_confidence,
    )


def test_sqlite_persists_events_across_repository_recreation_and_prevents_duplicates(tmp_path):
    path = tmp_path / "data" / "events.sqlite3"
    first = SQLiteRepository(path)
    first.initialize()

    assert first.save_event(_record()) is True
    assert first.save_event(_record()) is False
    second = SQLiteRepository(path)
    events = second.list_events()

    assert len(events) == 1
    assert events[0].event_id == "event-test"
    assert second.get_event("event-test").direction == "a_to_b"


def test_sqlite_round_trips_boundary_settings_and_generic_settings(tmp_path):
    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    repository.initialize()
    boundary = BoundaryConfig("camera-hash", "entrance", (0.1, 0.2), (0.9, 0.8),
                              BoundaryDirection.B_TO_A, True)

    repository.save_boundary(boundary)
    repository.set_setting("retention_days", None)

    assert repository.load_boundary("camera-hash", "entrance") == boundary
    assert repository.get_setting("retention_days", 30) is None
    assert repository.get_setting("missing", "default") == "default"


def test_identity_helpers_require_fields_and_support_removal(tmp_path):
    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    repository.initialize()
    repository.enroll_identity("id-1", "Consented Person", "/local/ref.jpg", consent_recorded=True)

    assert repository.list_identities()[0]["consent_recorded"] == 1
    assert repository.remove_identity("id-1") is True
    assert repository.remove_identity("id-1") is False
    with pytest.raises(ValueError):
        repository.enroll_identity("", "name", "/local/ref.jpg")


def test_image_store_writes_generated_paths_and_refuses_outside_delete(tmp_path):
    store = LocalImageStore(tmp_path / "images")
    image = np.full((64, 64, 3), 120, dtype=np.uint8)

    original = Path(store.save_face("event-1", image))
    clahe = Path(store.save_face("event-1", image, variant="clahe"))

    assert original.is_file() and clahe.is_file()
    assert original.name == "event-1-original.jpg"
    assert store.delete(original) is True
    with pytest.raises(ValueError, match="outside"):
        store.delete(tmp_path / "private.jpg")
    with pytest.raises(ValueError, match="characters"):
        store.save_face("../private", image)


def test_event_manager_saves_face_variants_and_event_metadata(tmp_path):
    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    manager = EventManager(repository, LocalImageStore(tmp_path / "images"))
    crop = np.random.default_rng(5).integers(40, 200, size=(96, 96, 3), dtype=np.uint8)
    quality = FaceQuality(96, 96, 120, 35, .92, {"minimum_dimensions": True})
    capture = FaceCaptureResult("event-test", 12, CaptureStatus.CAPTURED,
                                "quality_checks_passed", FaceDetection((0, 0, 96, 96), .92),
                                crop, quality)
    recognition = RecognitionResult(RecognitionStatus.NOT_RUN, "not_requested")

    result = manager.handle(_crossing(), capture, recognition)
    saved = repository.get_event("event-test")

    assert result.persisted is True
    assert saved.face_capture_status == "captured"
    assert saved.image_storage_status == "saved"
    assert Path(saved.original_face_path).is_file()
    assert Path(saved.clahe_face_path).is_file()
    assert saved.recognition_status == "not_run"
    assert manager.delete_event("event-test") is True
    assert not Path(saved.original_face_path).exists()
    assert repository.get_event("event-test") is None


def test_event_persists_even_without_face_or_when_alert_fails(tmp_path):
    class FailedAlert:
        def send(self, _event):
            raise RuntimeError("network down")

    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    manager = EventManager(repository, LocalImageStore(tmp_path / "images"), notifier=FailedAlert())
    capture = FaceCaptureResult("event-test", 12, CaptureStatus.NO_FACE, "no_face_detected")

    result = manager.handle(_crossing(), capture)

    assert result.persisted is True
    assert result.alert == AlertResult("failed", "alert_adapter_error")
    saved = repository.get_event("event-test")
    assert saved.face_capture_status == "no_face"
    assert saved.alert_status == "failed"
    assert "network down" not in (saved.alert_error or "")


def test_duplicate_event_does_not_send_a_second_alert(tmp_path):
    class CountAlert:
        def __init__(self):
            self.calls = 0

        def send(self, _event):
            self.calls += 1
            return AlertResult("sent")

    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    notifier = CountAlert()
    manager = EventManager(repository, LocalImageStore(tmp_path / "images"), notifier=notifier)

    first = manager.handle(_crossing())
    second = manager.handle(_crossing())

    assert first.persisted and not first.duplicate
    assert second.persisted and second.duplicate
    assert notifier.calls == 1


def test_image_write_failure_still_persists_crossing(tmp_path):
    class FailedImageStore(LocalImageStore):
        def save_face(self, *_args, **_kwargs):
            raise OSError("disk full")

    repository = SQLiteRepository(tmp_path / "events.sqlite3")
    manager = EventManager(repository, FailedImageStore(tmp_path / "images"))
    crop = np.random.default_rng(6).integers(40, 200, size=(96, 96, 3), dtype=np.uint8)
    quality = FaceQuality(96, 96, 120, 35, .92, {"sharpness": True})
    capture = FaceCaptureResult("event-test", 12, CaptureStatus.CAPTURED,
                                "quality_checks_passed", FaceDetection((0, 0, 96, 96), .92),
                                crop, quality)

    result = manager.handle(_crossing(), capture)

    assert result.persisted is True
    assert result.event.image_storage_status == "failed"
    assert repository.get_event("event-test").original_face_path is None


def test_event_manager_reports_database_failure_without_throwing(tmp_path):
    class FailedRepository(SQLiteRepository):
        def get_event(self, _event_id):
            return None

        def save_event(self, _event):
            raise DatabaseError("disk unavailable")

    manager = EventManager(FailedRepository(tmp_path / "events.sqlite3"),
                          LocalImageStore(tmp_path / "images"))
    result = manager.handle(_crossing())

    assert result.persisted is False
    assert "disk unavailable" in result.error


def test_telegram_notifier_skips_missing_configuration_and_handles_success_and_failure():
    skipped = TelegramNotifier("", "")
    assert skipped.send(_crossing()).status == "skipped"

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok":true}'

    sent = TelegramNotifier("secret-token", "chat", opener=lambda *_args, **_kwargs: Response())
    assert sent.send(_crossing()).status == "sent"
    failed = TelegramNotifier("secret-token", "chat",
                              opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
    failure = failed.send(_crossing())
    assert failure.status == "failed"
    assert "secret-token" not in str(failure)
