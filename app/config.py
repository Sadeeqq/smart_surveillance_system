"""Central application configuration with safe environment-based defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be a boolean value")
    return normalized in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Secrets remain optional and are never logged."""

    camera_source: str
    camera_reconnect_attempts: int
    model_name: str
    confidence_threshold: float
    detection_image_size: int
    detection_device: str
    tracking_confidence_threshold: float
    tracker_config: str
    tracking_trail_length: int
    storage_dir: Path
    event_cooldown_seconds: int
    telegram_enabled: bool
    telegram_bot_token: str | None
    telegram_chat_id: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        """Read SSS_* variables and optional local .env; missing values default safely."""
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        confidence = _read_float("SSS_CONFIDENCE_THRESHOLD", 0.5)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("SSS_CONFIDENCE_THRESHOLD must be between 0 and 1")
        image_size = _read_int("SSS_DETECTION_IMAGE_SIZE", 640)
        if image_size < 32:
            raise ValueError("SSS_DETECTION_IMAGE_SIZE must be at least 32")
        tracking_confidence = _read_float("SSS_TRACKING_CONFIDENCE_THRESHOLD", 0.1)
        if not 0.0 <= tracking_confidence <= 1.0:
            raise ValueError("SSS_TRACKING_CONFIDENCE_THRESHOLD must be between 0 and 1")
        tracking_trail_length = _read_int("SSS_TRACKING_TRAIL_LENGTH", 30)
        if tracking_trail_length < 1:
            raise ValueError("SSS_TRACKING_TRAIL_LENGTH must be at least 1")
        cooldown = _read_int("SSS_EVENT_COOLDOWN_SECONDS", 5)
        if cooldown < 0:
            raise ValueError("SSS_EVENT_COOLDOWN_SECONDS cannot be negative")
        reconnect_attempts = _read_int("SSS_CAMERA_RECONNECT_ATTEMPTS", 3)
        if reconnect_attempts < 0:
            raise ValueError("SSS_CAMERA_RECONNECT_ATTEMPTS cannot be negative")

        return cls(
            camera_source=os.getenv("SSS_CAMERA_SOURCE", "0").strip() or "0",
            camera_reconnect_attempts=reconnect_attempts,
            model_name=os.getenv("SSS_MODEL_NAME", "yolo11n.pt").strip() or "yolo11n.pt",
            confidence_threshold=confidence,
            detection_image_size=image_size,
            detection_device=os.getenv("SSS_DETECTION_DEVICE", "cpu").strip() or "cpu",
            tracking_confidence_threshold=tracking_confidence,
            tracker_config=os.getenv("SSS_TRACKER_CONFIG", "bytetrack.yaml").strip() or "bytetrack.yaml",
            tracking_trail_length=tracking_trail_length,
            storage_dir=Path(os.getenv("SSS_STORAGE_DIR", "data")).expanduser(),
            event_cooldown_seconds=cooldown,
            telegram_enabled=_read_bool("SSS_TELEGRAM_ENABLED", False),
            telegram_bot_token=os.getenv("SSS_TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.getenv("SSS_TELEGRAM_CHAT_ID") or None,
        )
