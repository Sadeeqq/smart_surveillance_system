"""Tests for phase 0 configuration defaults and validation."""

import pytest

from app.config import Settings


def test_optional_environment_values_use_safe_defaults(monkeypatch):
    for name in (
        "SSS_CAMERA_SOURCE", "SSS_MODEL_NAME", "SSS_CONFIDENCE_THRESHOLD",
        "SSS_STORAGE_DIR", "SSS_EVENT_COOLDOWN_SECONDS", "SSS_TELEGRAM_ENABLED",
        "SSS_TELEGRAM_BOT_TOKEN", "SSS_TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.camera_source == "0"
    assert settings.confidence_threshold == 0.5
    assert settings.event_cooldown_seconds == 5
    assert settings.telegram_enabled is False
    assert settings.telegram_bot_token is None


def test_invalid_confidence_is_rejected(monkeypatch):
    monkeypatch.setenv("SSS_CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings.from_env()


def test_missing_telegram_settings_do_not_enable_notifications(monkeypatch):
    monkeypatch.delenv("SSS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SSS_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("SSS_TELEGRAM_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.telegram_enabled is False
    assert settings.telegram_chat_id is None


def test_local_dotenv_is_loaded_without_overriding_process_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SSS_CAMERA_SOURCE", "7")
    (tmp_path / ".env").write_text(
        "SSS_CAMERA_SOURCE=2\nSSS_EVENT_COOLDOWN_SECONDS=12\n",
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.camera_source == "7"
    assert settings.event_cooldown_seconds == 12
