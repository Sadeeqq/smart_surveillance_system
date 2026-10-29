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
    model_name: str
    confidence_threshold: float
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
        cooldown = _read_int("SSS_EVENT_COOLDOWN_SECONDS", 5)
        if cooldown < 0:
            raise ValueError("SSS_EVENT_COOLDOWN_SECONDS cannot be negative")

        return cls(
            camera_source=os.getenv("SSS_CAMERA_SOURCE", "0").strip() or "0",
            model_name=os.getenv("SSS_MODEL_NAME", "yolo11n.pt").strip() or "yolo11n.pt",
            confidence_threshold=confidence,
            storage_dir=Path(os.getenv("SSS_STORAGE_DIR", "data")).expanduser(),
            event_cooldown_seconds=cooldown,
            telegram_enabled=_read_bool("SSS_TELEGRAM_ENABLED", False),
            telegram_bot_token=os.getenv("SSS_TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.getenv("SSS_TELEGRAM_CHAT_ID") or None,
        )
