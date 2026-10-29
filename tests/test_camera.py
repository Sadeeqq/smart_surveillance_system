"""Camera adapter tests using fake captures and a generated video file."""

import cv2
import numpy as np
import pytest

from app.camera import CameraState, OpenCVCameraSource


class FakeCapture:
    def __init__(self, opened=True, reads=None):
        self.opened = opened
        self.reads = list(reads or [])
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self):
        self.released = True


def test_connect_read_and_release():
    frame = object()
    capture = FakeCapture(reads=[(True, frame)])
    camera = OpenCVCameraSource("0", capture_factory=lambda source: capture)

    assert camera.connect() is True
    assert camera.status().state is CameraState.CONNECTED
    assert camera.read_frame() is frame
    camera.release()
    camera.release()
    assert capture.released is True
    assert camera.status().state is CameraState.DISCONNECTED


def test_failed_open_reports_error_and_releases_capture():
    capture = FakeCapture(opened=False)
    camera = OpenCVCameraSource(0, capture_factory=lambda source: capture)

    assert camera.connect() is False
    assert camera.status().state is CameraState.ERROR
    assert "check the device" in camera.status().message
    assert capture.released is True


def test_stream_loss_reconnects_once_and_returns_no_stale_frame():
    broken = FakeCapture(reads=[(False, None)])
    recovered = FakeCapture(reads=[(True, object())])
    captures = iter([broken, recovered])
    camera = OpenCVCameraSource(
        "video.mp4",
        reconnect_attempts=2,
        reconnect_delay_seconds=0,
        capture_factory=lambda source: next(captures),
    )

    assert camera.connect() is True
    assert camera.read_frame() is None
    assert camera.status().state is CameraState.CONNECTED
    assert camera.status().reconnect_attempt == 1
    assert broken.released is True
    assert camera.read_frame() is not None


def test_reconnect_attempts_are_bounded():
    opened = FakeCapture(reads=[(False, None)])
    closed = FakeCapture(opened=False)
    captures = iter([opened, closed, closed])
    camera = OpenCVCameraSource(
        0,
        reconnect_attempts=2,
        reconnect_delay_seconds=0,
        capture_factory=lambda source: next(captures),
    )

    assert camera.connect() is True
    assert camera.read_frame() is None
    assert camera.status().state is CameraState.ERROR
    assert camera.status().reconnect_attempt == 2


def test_reads_frames_from_a_recorded_video(tmp_path):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable in this OpenCV build")
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    camera = OpenCVCameraSource(str(video_path), reconnect_attempts=0)
    try:
        assert camera.connect() is True
        frame = camera.read_frame()
        assert frame is not None
        assert frame.shape == (24, 32, 3)
    finally:
        camera.release()
