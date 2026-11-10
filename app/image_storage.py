"""Local event-image storage with generated, non-sensitive filenames."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2


class ImageStorageError(RuntimeError):
    """Raised when an event image cannot safely be encoded or written."""


class LocalImageStore:
    """Write image crops beneath a configured root, never using user names in paths."""

    _SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

    def __init__(self, root: str | Path = "data/event_images", *, jpeg_quality: int = 92) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.root = Path(root).expanduser().resolve()
        self.jpeg_quality = jpeg_quality

    def save_face(self, event_id: str, image: Any, *, variant: str = "original") -> str:
        if not self._SAFE_ID.fullmatch(event_id):
            raise ValueError("event_id contains unsupported filename characters")
        if variant not in {"original", "clahe"}:
            raise ValueError("variant must be 'original' or 'clahe'")
        if image is None or getattr(image, "size", 1) == 0:
            raise ImageStorageError("cannot store an empty face image")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ImageStorageError("OpenCV could not encode the face image")
        path = self.root / f"{event_id}-{variant}.jpg"
        temp_name: str | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=self.root, prefix=f".{event_id}-",
                                             suffix=".tmp", delete=False) as handle:
                temp_name = handle.name
                handle.write(encoded.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
            temp_name = None
            try:
                return str(path.relative_to(Path.cwd().resolve()))
            except ValueError:
                return str(path)
        except OSError as exc:
            raise ImageStorageError(f"Could not write {variant} face image: {exc}") from exc
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def delete(self, stored_path: str | Path | None) -> bool:
        """Delete only a file resolved under this image store's root."""
        if not stored_path:
            return False
        path = Path(stored_path).expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("refusing to delete a path outside the configured image store") from exc
        if path.is_file():
            path.unlink()
            return True
        return False
