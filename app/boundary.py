"""Normalized virtual boundaries and direction-aware crossing detection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class BoundaryDirection(str, Enum):
    """Direction across an oriented A-to-B line.

    A_TO_B means the anchor moves from the negative signed side to the
    positive signed side. B_TO_A is the reverse. The sign is calculated from
    the oriented endpoints in image coordinates and is stable across sizes.
    """

    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """One camera's editable line, stored in normalized frame coordinates."""

    camera_id: str
    boundary_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    direction_mode: BoundaryDirection = BoundaryDirection.BOTH
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or not self.boundary_id.strip():
            raise ValueError("camera_id and boundary_id cannot be empty")
        for label, point in (("start", self.start), ("end", self.end)):
            if len(point) != 2 or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in point):
                raise ValueError(f"{label} must contain two finite normalized coordinates")
        if self.start == self.end:
            raise ValueError("boundary endpoints must be different")
        if not isinstance(self.direction_mode, BoundaryDirection):
            object.__setattr__(self, "direction_mode", BoundaryDirection(self.direction_mode))


@dataclass(frozen=True, slots=True)
class BoundaryCrossingEvent:
    """A confirmed crossing; this is an event candidate for later persistence."""

    event_id: str
    timestamp: datetime
    camera_id: str
    track_id: int
    boundary_id: str
    direction: BoundaryDirection
    detection_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "boundary_id": self.boundary_id,
            "direction": self.direction.value,
            "detection_confidence": self.detection_confidence,
        }


def camera_identifier(source: str | int) -> str:
    """Return a stable, non-revealing identifier for a camera source string."""

    return hashlib.sha256(str(source).strip().encode("utf-8")).hexdigest()[:16]


def signed_distance_to_boundary(
    point: tuple[float, float],
    boundary: BoundaryConfig,
    frame_size: tuple[int, int],
) -> float:
    """Return signed pixel distance from a frame-relative point to the line.

    ``frame_size`` is ``(width, height)``. Normalized endpoints are mapped to
    the actual frame dimensions before geometry is evaluated, so resized video
    preserves the same relative boundary placement.
    """

    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame width and height must be positive")
    ax, ay = boundary.start[0] * width, boundary.start[1] * height
    bx, by = boundary.end[0] * width, boundary.end[1] * height
    px, py = point
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("boundary endpoints collapse at this frame size")
    return (dx * (py - ay) - dy * (px - ax)) / length


def _side(distance: float, neutral_distance: float) -> int:
    if distance > neutral_distance:
        return 1
    if distance < -neutral_distance:
        return -1
    return 0


class BoundaryStore:
    """Small atomic JSON store used until Phase 6 introduces SQLite settings."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = "data/boundaries.json") -> None:
        self.path = Path(path).expanduser()

    def load(self, camera_id: str, boundary_id: str = "default") -> BoundaryConfig | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("boundary settings root must be an object")
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("unsupported boundary settings schema version")
            cameras = payload.get("boundaries", {})
            if not isinstance(cameras, dict):
                raise ValueError("boundary settings must contain a boundaries object")
            camera_boundaries = cameras.get(camera_id, {})
            if not isinstance(camera_boundaries, dict):
                raise ValueError("camera boundary settings must be an object")
            raw = camera_boundaries.get(boundary_id)
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise ValueError("saved boundary must be an object")
            return BoundaryConfig(
                camera_id=camera_id,
                boundary_id=boundary_id,
                start=tuple(raw["start"]),
                end=tuple(raw["end"]),
                direction_mode=BoundaryDirection(raw["direction_mode"]),
                enabled=bool(raw["enabled"]),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Could not read boundary settings {self.path}: {exc}") from exc

    def save(self, boundary: BoundaryConfig) -> None:
        payload: dict[str, Any] = {"schema_version": self.SCHEMA_VERSION, "boundaries": {}}
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("boundary settings root must be an object")
                if payload.get("schema_version") != self.SCHEMA_VERSION:
                    raise ValueError("unsupported boundary settings schema version")
                if not isinstance(payload.get("boundaries"), dict):
                    raise ValueError("boundary settings must contain a boundaries object")
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read boundary settings {self.path}: {exc}") from exc
        camera_boundaries = payload.setdefault("boundaries", {}).setdefault(boundary.camera_id, {})
        camera_boundaries[boundary.boundary_id] = {
            "start": list(boundary.start),
            "end": list(boundary.end),
            "direction_mode": boundary.direction_mode.value,
            "enabled": boundary.enabled,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, self.path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


@dataclass(slots=True)
class _TrackState:
    last_seen_frame: int
    stable_side: int | None = None
    candidate_side: int | None = None
    candidate_observations: int = 0
    last_event_at: datetime | None = None


class BoundaryEngine:
    """Debounce tracked-person side changes into direction-aware events."""

    def __init__(
        self,
        *,
        neutral_zone_fraction: float = 0.01,
        min_track_observations: int = 2,
        crossing_confirm_frames: int = 2,
        cooldown_seconds: float = 5.0,
        max_missed_frames: int = 30,
    ) -> None:
        if not 0.0 <= neutral_zone_fraction < 0.5:
            raise ValueError("neutral_zone_fraction must be in [0, 0.5)")
        if min_track_observations < 1 or crossing_confirm_frames < 1 or max_missed_frames < 1:
            raise ValueError("frame observation limits must be positive")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.neutral_zone_fraction = neutral_zone_fraction
        self.min_track_observations = min_track_observations
        self.crossing_confirm_frames = crossing_confirm_frames
        self.cooldown_seconds = cooldown_seconds
        self.max_missed_frames = max_missed_frames
        self._frame_index = 0
        self._tracks: dict[tuple[str, str, int], _TrackState] = {}

    def reset(self) -> None:
        """Clear in-memory crossing state, for example after a source restart."""

        self._tracks.clear()
        self._frame_index = 0

    def update(
        self,
        boundary: BoundaryConfig,
        frame_size: tuple[int, int],
        tracked_people: Iterable[Any],
        *,
        timestamp: datetime | None = None,
    ) -> list[BoundaryCrossingEvent]:
        """Observe one frame of tracked people and return confirmed crossings."""

        self._frame_index += 1
        now = timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        width, height = frame_size
        neutral_distance = self.neutral_zone_fraction * math.hypot(width, height)
        events: list[BoundaryCrossingEvent] = []
        seen: set[tuple[str, str, int]] = set()

        for person in tracked_people:
            key = (boundary.camera_id, boundary.boundary_id, int(person.track_id))
            seen.add(key)
            state = self._tracks.setdefault(key, _TrackState(last_seen_frame=self._frame_index))
            state.last_seen_frame = self._frame_index
            anchor = person.bottom_center
            current_side = _side(signed_distance_to_boundary(anchor, boundary, frame_size), neutral_distance)

            if state.stable_side is None:
                self._learn_initial_side(state, current_side)
                continue
            if current_side == 0 or current_side == state.stable_side:
                state.candidate_side = None
                state.candidate_observations = 0
                continue
            if current_side != state.candidate_side:
                state.candidate_side = current_side
                state.candidate_observations = 1
            else:
                state.candidate_observations += 1
            if state.candidate_observations < self.crossing_confirm_frames:
                continue

            direction = (
                BoundaryDirection.A_TO_B
                if state.stable_side == -1 and current_side == 1
                else BoundaryDirection.B_TO_A
            )
            previous_side = state.stable_side
            state.stable_side = current_side
            state.candidate_side = None
            state.candidate_observations = 0
            if not boundary.enabled or not self._direction_allowed(boundary.direction_mode, direction):
                continue
            if state.last_event_at is not None and (now - state.last_event_at).total_seconds() < self.cooldown_seconds:
                continue

            event = BoundaryCrossingEvent(
                event_id=str(uuid.uuid4()),
                timestamp=now,
                camera_id=boundary.camera_id,
                track_id=int(person.track_id),
                boundary_id=boundary.boundary_id,
                direction=direction,
                detection_confidence=float(person.confidence),
            )
            events.append(event)
            state.last_event_at = now

        for key, state in list(self._tracks.items()):
            if key not in seen and self._frame_index - state.last_seen_frame > self.max_missed_frames:
                del self._tracks[key]
        return events

    def _learn_initial_side(self, state: _TrackState, current_side: int) -> None:
        if current_side == 0:
            state.candidate_side = None
            state.candidate_observations = 0
            return
        if current_side != state.candidate_side:
            state.candidate_side = current_side
            state.candidate_observations = 1
        else:
            state.candidate_observations += 1
        if state.candidate_observations >= self.min_track_observations:
            state.stable_side = current_side
            state.candidate_side = None
            state.candidate_observations = 0

    @staticmethod
    def _direction_allowed(mode: BoundaryDirection, direction: BoundaryDirection) -> bool:
        return mode is BoundaryDirection.BOTH or mode is direction
