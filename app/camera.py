"""Replaceable OpenCV camera sources for webcams, virtual cameras and video files."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import cv2

logger = logging.getLogger(__name__)


class CameraState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CameraStatus:
    state: CameraState
    message: str
    reconnect_attempt: int = 0


class CameraSource(Protocol):
    """Contract consumed by the processing pipeline."""

    def connect(self) -> bool: ...

    def read_frame(self) -> Any | None: ...

    def status(self) -> CameraStatus: ...

    def release(self) -> None: ...


def _parse_source(source: str | int) -> str | int:
    if isinstance(source, int):
        if source < 0:
            raise ValueError("Camera index cannot be negative")
        return source
    source = source.strip()
    if not source:
        raise ValueError("Camera source cannot be empty")
    if source.isdecimal():
        return int(source)
    return source


class OpenCVCameraSource:
    """OpenCV adapter with explicit status and bounded reconnect attempts.

    ``source`` may be a local camera index, a virtual camera device path,
    an OpenCV-compatible stream URL, or a recorded video path.
    """

    def __init__(
        self,
        source: str | int,
        *,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 0.5,
        capture_factory: Callable[[str | int], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts cannot be negative")
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds cannot be negative")
        self._source = _parse_source(source)
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay_seconds
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._sleep = sleep
        self._capture: Any | None = None
        self._status = CameraStatus(CameraState.DISCONNECTED, "Camera is not connected")

    def connect(self) -> bool:
        """Open the configured source and report a useful failure state."""
        self.release()
        self._status = CameraStatus(CameraState.CONNECTING, "Connecting to camera")
        try:
            capture = self._capture_factory(self._source)
            if not capture.isOpened():
                capture.release()
                self._status = CameraStatus(
                    CameraState.ERROR,
                    "Unable to open camera source; check the device, stream URL, permissions, and connection",
                )
                return False
            self._capture = capture
            self._status = CameraStatus(CameraState.CONNECTED, "Camera connected")
            return True
        except Exception as exc:  # OpenCV backends can raise platform-specific errors.
            self._status = CameraStatus(CameraState.ERROR, f"Camera connection failed: {exc}")
            logger.exception("Camera connection failed")
            return False

    def read_frame(self) -> Any | None:
        """Read a frame, attempting a bounded reconnect after disconnect."""
        if self._capture is None:
            return None
        try:
            ok, frame = self._capture.read()
        except Exception:
            logger.exception("Camera frame read failed")
            ok, frame = False, None
        if ok and frame is not None:
            self._status = CameraStatus(CameraState.CONNECTED, "Camera connected")
            return frame

        logger.warning("Camera returned an empty frame; attempting bounded reconnect")
        self._close_capture()
        self._status = CameraStatus(CameraState.DISCONNECTED, "Camera stream lost; reconnecting")
        for attempt in range(1, self._reconnect_attempts + 1):
            if self._reconnect_delay:
                self._sleep(self._reconnect_delay * attempt)
            self._status = CameraStatus(
                CameraState.CONNECTING,
                "Reconnecting to camera",
                reconnect_attempt=attempt,
            )
            try:
                capture = self._capture_factory(self._source)
                if not capture.isOpened():
                    capture.release()
                    continue
                self._capture = capture
                self._status = CameraStatus(
                    CameraState.CONNECTED,
                    "Camera reconnected; waiting for next frame",
                    reconnect_attempt=attempt,
                )
                return None
            except Exception:
                logger.exception("Camera reconnect attempt %d failed", attempt)
        self._status = CameraStatus(
            CameraState.ERROR,
            "Camera stream unavailable after bounded reconnect attempts; call connect() to retry",
            reconnect_attempt=self._reconnect_attempts,
        )
        return None

    def status(self) -> CameraStatus:
        """Return the latest connection state without exposing source credentials."""
        return self._status

    def _close_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.exception("Failed to release camera capture")

    def release(self) -> None:
        """Release the device; safe to call repeatedly."""
        self._close_capture()
        self._status = CameraStatus(CameraState.DISCONNECTED, "Camera released")
