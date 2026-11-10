"""SQLite repositories for crossing events, boundaries, identities and settings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from app.boundary import BoundaryConfig, BoundaryDirection


class DatabaseError(RuntimeError):
    """Raised when a persistence operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    timestamp: str
    camera_id: str
    track_id: int
    boundary_id: str
    direction: str
    detection_confidence: float
    face_capture_status: str = "not_attempted"
    face_capture_reason: str | None = None
    face_quality_json: str | None = None
    recognition_status: str = "not_run"
    recognition_reason: str | None = None
    matched_identity: str | None = None
    recognition_distance: float | None = None
    recognition_threshold: float | None = None
    original_face_path: str | None = None
    clahe_face_path: str | None = None
    image_storage_status: str = "not_attempted"
    alert_status: str = "disabled"
    alert_error: str | None = None
    reviewed: bool = False

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.camera_id.strip() or not self.boundary_id.strip():
            raise ValueError("event_id, camera_id and boundary_id cannot be empty")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0 and 1")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EventRecord":
        values = dict(row)
        values["reviewed"] = bool(values.get("reviewed", False))
        return cls(**values)


class SQLiteRepository:
    """Small connection-per-operation SQLite repository with idempotent event inserts."""

    def __init__(self, path: str | Path = "data/events.sqlite3") -> None:
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY,
            display_name TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS boundaries (
            camera_id TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
            boundary_id TEXT NOT NULL,
            start_x REAL NOT NULL,
            start_y REAL NOT NULL,
            end_x REAL NOT NULL,
            end_y REAL NOT NULL,
            direction_mode TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (camera_id, boundary_id)
        );
        CREATE TABLE IF NOT EXISTS enrolled_identities (
            identity_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            reference_image_path TEXT NOT NULL,
            consent_recorded INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS settings (
            setting_key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
            track_id INTEGER NOT NULL,
            boundary_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            detection_confidence REAL NOT NULL,
            face_capture_status TEXT NOT NULL,
            face_capture_reason TEXT,
            face_quality_json TEXT,
            recognition_status TEXT NOT NULL,
            recognition_reason TEXT,
            matched_identity TEXT,
            recognition_distance REAL,
            recognition_threshold REAL,
            original_face_path TEXT,
            clahe_face_path TEXT,
            image_storage_status TEXT NOT NULL,
            alert_status TEXT NOT NULL,
            alert_error TEXT,
            reviewed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_events_camera_timestamp ON events(camera_id, timestamp DESC);
        """
        try:
            with self._connect() as connection:
                connection.executescript(schema)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not initialize SQLite database {self.path}: {exc}") from exc

    def save_event(self, event: EventRecord) -> bool:
        """Insert once by event ID. Return False for a duplicate rather than alert twice."""
        values = asdict(event)
        values["reviewed"] = int(event.reviewed)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO cameras(camera_id) VALUES (?)", (event.camera_id,)
                )
                cursor = connection.execute(
                    f"INSERT OR IGNORE INTO events ({columns}) VALUES ({placeholders})",
                    tuple(values.values()),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not save event {event.event_id}: {exc}") from exc

    def update_event_delivery(self, event_id: str, *, alert_status: str,
                              alert_error: str | None = None) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE events SET alert_status=?, alert_error=? WHERE event_id=?",
                    (alert_status, alert_error, event_id),
                )
                if cursor.rowcount != 1:
                    raise DatabaseError(f"Event {event_id} does not exist")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not update delivery status for event {event_id}: {exc}") from exc

    def get_event(self, event_id: str) -> EventRecord | None:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
                return EventRecord.from_mapping(dict(row)) if row else None
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise DatabaseError(f"Could not read event {event_id}: {exc}") from exc

    def list_events(self, *, limit: int = 100, offset: int = 0,
                    camera_id: str | None = None) -> list[EventRecord]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset cannot be negative")
        query = "SELECT * FROM events"
        parameters: list[Any] = []
        if camera_id is not None:
            query += " WHERE camera_id=?"
            parameters.append(camera_id)
        query += " ORDER BY timestamp DESC, event_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        try:
            with self._connect() as connection:
                rows = connection.execute(query, parameters).fetchall()
                return [EventRecord.from_mapping(dict(row)) for row in rows]
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise DatabaseError(f"Could not list events: {exc}") from exc

    def save_boundary(self, boundary: BoundaryConfig) -> None:
        try:
            with self._connect() as connection:
                connection.execute("INSERT OR IGNORE INTO cameras(camera_id) VALUES (?)",
                                   (boundary.camera_id,))
                connection.execute(
                    """INSERT INTO boundaries(camera_id,boundary_id,start_x,start_y,end_x,end_y,
                       direction_mode,enabled,updated_at) VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                       ON CONFLICT(camera_id,boundary_id) DO UPDATE SET
                       start_x=excluded.start_x,start_y=excluded.start_y,end_x=excluded.end_x,
                       end_y=excluded.end_y,direction_mode=excluded.direction_mode,
                       enabled=excluded.enabled,updated_at=CURRENT_TIMESTAMP""",
                    (boundary.camera_id, boundary.boundary_id, *boundary.start, *boundary.end,
                     boundary.direction_mode.value, int(boundary.enabled)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not save boundary {boundary.boundary_id}: {exc}") from exc

    def load_boundary(self, camera_id: str, boundary_id: str = "default") -> BoundaryConfig | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM boundaries WHERE camera_id=? AND boundary_id=?",
                    (camera_id, boundary_id),
                ).fetchone()
            if row is None:
                return None
            return BoundaryConfig(
                camera_id=row["camera_id"], boundary_id=row["boundary_id"],
                start=(row["start_x"], row["start_y"]), end=(row["end_x"], row["end_y"]),
                direction_mode=BoundaryDirection(row["direction_mode"]), enabled=bool(row["enabled"]),
            )
        except (sqlite3.Error, ValueError) as exc:
            raise DatabaseError(f"Could not load boundary for camera {camera_id}: {exc}") from exc

    def set_setting(self, key: str, value: Any) -> None:
        if not key.strip():
            raise ValueError("setting key cannot be empty")
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO settings(setting_key,value_json) VALUES(?,?)
                       ON CONFLICT(setting_key) DO UPDATE SET value_json=excluded.value_json,
                       updated_at=CURRENT_TIMESTAMP""", (key, encoded),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not save setting {key}: {exc}") from exc

    def get_setting(self, key: str, default: Any = None) -> Any:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT value_json FROM settings WHERE setting_key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise DatabaseError(f"Could not read setting {key}: {exc}") from exc

    def enroll_identity(self, identity_id: str, display_name: str, reference_image_path: str,
                        *, consent_recorded: bool = False) -> None:
        if not all(value.strip() for value in (identity_id, display_name, reference_image_path)):
            raise ValueError("identity ID, display name and reference image path are required")
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO enrolled_identities(identity_id,display_name,reference_image_path,
                       consent_recorded) VALUES(?,?,?,?) ON CONFLICT(identity_id) DO UPDATE SET
                       display_name=excluded.display_name,reference_image_path=excluded.reference_image_path,
                       consent_recorded=excluded.consent_recorded""",
                    (identity_id, display_name, reference_image_path, int(consent_recorded)),
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not store enrolled identity {identity_id}: {exc}") from exc

    def list_identities(self) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT identity_id,display_name,reference_image_path,consent_recorded,created_at "
                    "FROM enrolled_identities ORDER BY display_name"
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not list enrolled identities: {exc}") from exc

    def remove_identity(self, identity_id: str) -> bool:
        try:
            with self._connect() as connection:
                cursor = connection.execute("DELETE FROM enrolled_identities WHERE identity_id=?", (identity_id,))
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not remove enrolled identity {identity_id}: {exc}") from exc

    def delete_event(self, event_id: str) -> EventRecord | None:
        """Delete metadata and return it so a caller can remove referenced files explicitly."""
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
                if row is None:
                    return None
                record = EventRecord.from_mapping(dict(row))
                connection.execute("DELETE FROM events WHERE event_id=?", (event_id,))
                return record
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise DatabaseError(f"Could not delete event {event_id}: {exc}") from exc
