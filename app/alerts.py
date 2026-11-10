"""Optional notification adapters. Local event persistence does not depend on them."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.boundary import BoundaryCrossingEvent


@dataclass(frozen=True, slots=True)
class AlertResult:
    status: str
    reason: str | None = None


class AlertAdapter(Protocol):
    def send(self, event: BoundaryCrossingEvent) -> AlertResult: ...


class TelegramNotifier:
    """Send a text-only crossing summary through the Telegram Bot API."""

    def __init__(self, bot_token: str | None, chat_id: str | None, *, timeout_seconds: float = 8.0,
                 opener=urllib.request.urlopen) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._bot_token = bot_token.strip() if bot_token else ""
        self._chat_id = chat_id.strip() if chat_id else ""
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    @property
    def configured(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    def send(self, event: BoundaryCrossingEvent) -> AlertResult:
        if not self.configured:
            return AlertResult("skipped", "telegram_not_configured")
        message = (
            "Smart Surveillance crossing event\n"
            f"Time: {event.timestamp.isoformat()}\n"
            f"Camera: {event.camera_id}\n"
            f"Boundary: {event.boundary_id}\n"
            f"Direction: {event.direction.value}\n"
            f"Detection confidence: {event.detection_confidence:.2f}"
        )
        payload = json.dumps({"chat_id": self._chat_id, "text": message}).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and body.get("ok") is True:
                    return AlertResult("sent")
                return AlertResult("failed", "telegram_api_rejected_message")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            # Never include the request URL or exception text, because the URL embeds the bot token.
            return AlertResult("failed", "telegram_request_failed")

