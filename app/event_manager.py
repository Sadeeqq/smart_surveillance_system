"""Coordinate event enrichment, local image writes, SQLite and optional alerts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from app.alerts import AlertAdapter, AlertResult
from app.boundary import BoundaryCrossingEvent
from app.database import DatabaseError, EventRecord, SQLiteRepository
from app.face_processing import (
    CaptureStatus,
    FaceCaptureResult,
    FaceImageProcessor,
    RecognitionResult,
    RecognitionStatus,
)
from app.image_storage import ImageStorageError, LocalImageStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EventHandlingResult:
    event: EventRecord
    persisted: bool
    duplicate: bool = False
    alert: AlertResult = AlertResult("disabled")
    error: str | None = None


class EventManager:
    """Save every crossing event first; optional image/notification failures are isolated."""

    def __init__(self, repository: SQLiteRepository, image_store: LocalImageStore,
                 *, notifier: AlertAdapter | None = None,
                 image_processor: FaceImageProcessor | None = None) -> None:
        self.repository = repository
        self.image_store = image_store
        self.notifier = notifier
        self.image_processor = image_processor or FaceImageProcessor()
        self.repository.initialize()

    def handle(self, crossing: BoundaryCrossingEvent,
               face_capture: FaceCaptureResult | None = None,
               recognition: RecognitionResult | None = None) -> EventHandlingResult:
        try:
            existing = self.repository.get_event(crossing.event_id)
        except DatabaseError as exc:
            logger.warning("Could not check for duplicate event %s: %s", crossing.event_id, exc)
            existing = None
        if existing is not None:
            return EventHandlingResult(existing, persisted=True, duplicate=True,
                                       alert=AlertResult(existing.alert_status, "duplicate_event"))

        original_path = clahe_path = None
        image_status = "not_attempted"
        capture_status = face_capture.status.value if face_capture else "not_attempted"
        capture_reason = face_capture.reason if face_capture else None
        quality_json = None
        if face_capture and face_capture.quality:
            quality_json = json.dumps({
                "width": face_capture.quality.width,
                "height": face_capture.quality.height,
                "brightness": face_capture.quality.brightness,
                "laplacian_variance": face_capture.quality.blur_score,
                "face_confidence": face_capture.quality.face_confidence,
                "checks": dict(face_capture.quality.checks),
            }, separators=(",", ":"), sort_keys=True)

        if face_capture and face_capture.status is CaptureStatus.CAPTURED and face_capture.original_crop is not None:
            try:
                variants = self.image_processor.variants(face_capture.original_crop)
                original_path = self.image_store.save_face(crossing.event_id, variants.original,
                                                           variant="original")
                clahe_path = self.image_store.save_face(crossing.event_id, variants.clahe,
                                                       variant="clahe")
                image_status = "saved"
            except Exception as exc:
                image_status = "failed"
                capture_reason = f"{capture_reason or 'face_captured'};image_storage_failed"
                logger.warning("Face image storage failed for event %s: %s", crossing.event_id, exc)
                for stored_path in (original_path, clahe_path):
                    try:
                        self.image_store.delete(stored_path)
                    except (OSError, ValueError):
                        logger.exception("Could not clean partial event images for %s", crossing.event_id)
                original_path = clahe_path = None

        recognition = recognition or RecognitionResult(RecognitionStatus.NOT_RUN, "recognition_not_requested")
        alert_status = "pending" if self.notifier is not None else "disabled"
        record = EventRecord(
            event_id=crossing.event_id,
            timestamp=crossing.timestamp.isoformat(),
            camera_id=crossing.camera_id,
            track_id=crossing.track_id,
            boundary_id=crossing.boundary_id,
            direction=crossing.direction.value,
            detection_confidence=crossing.detection_confidence,
            face_capture_status=capture_status,
            face_capture_reason=capture_reason,
            face_quality_json=quality_json,
            recognition_status=recognition.status.value,
            recognition_reason=recognition.reason,
            matched_identity=recognition.identity,
            recognition_distance=recognition.distance,
            recognition_threshold=recognition.threshold,
            original_face_path=original_path,
            clahe_face_path=clahe_path,
            image_storage_status=image_status,
            alert_status=alert_status,
        )
        try:
            inserted = self.repository.save_event(record)
        except DatabaseError as exc:
            logger.error("Crossing event %s could not be persisted: %s", crossing.event_id, exc)
            for stored_path in (original_path, clahe_path):
                try:
                    self.image_store.delete(stored_path)
                except (OSError, ValueError):
                    logger.exception("Could not clean orphan event image for %s", crossing.event_id)
            return EventHandlingResult(record, persisted=False, error=str(exc))

        if not inserted:
            saved = self.repository.get_event(crossing.event_id) or record
            return EventHandlingResult(saved, persisted=True, duplicate=True,
                                       alert=AlertResult(saved.alert_status, "duplicate_event"))

        alert_result = AlertResult("disabled")
        if self.notifier is not None:
            try:
                alert_result = self.notifier.send(crossing)
            except Exception as exc:  # Alert plugins must never break local event storage.
                logger.warning("Alert adapter failed for event %s: %s", crossing.event_id, exc)
                alert_result = AlertResult("failed", "alert_adapter_error")
            try:
                self.repository.update_event_delivery(
                    crossing.event_id, alert_status=alert_result.status,
                    alert_error=alert_result.reason if alert_result.status == "failed" else None,
                )
            except DatabaseError as exc:
                logger.error("Could not record alert status for event %s: %s", crossing.event_id, exc)
                return EventHandlingResult(record, persisted=True, alert=alert_result,
                                           error="event_saved_but_alert_status_update_failed")
            record = replace(record, alert_status=alert_result.status,
                             alert_error=alert_result.reason if alert_result.status == "failed" else None)
        return EventHandlingResult(record, persisted=True, alert=alert_result)

    def delete_event(self, event_id: str) -> bool:
        """Delete event metadata and its referenced local face files, if present."""
        record = self.repository.delete_event(event_id)
        if record is None:
            return False
        for image_path in (record.original_face_path, record.clahe_face_path):
            try:
                self.image_store.delete(image_path)
            except (OSError, ValueError):
                logger.exception("Event %s was deleted but an image could not be removed", event_id)
        return True
