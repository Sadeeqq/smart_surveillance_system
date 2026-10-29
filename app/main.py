"""Phase 0 entry point; starts without camera/model dependencies."""

import logging

from app.config import Settings


def configure_logging() -> None:
    """Set a concise, consistent logging format for the prototype."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    configure_logging()
    settings = Settings.from_env()
    logging.getLogger(__name__).info(
        "Smart Surveillance System foundation ready (storage=%s, camera_source=%s)",
        settings.storage_dir,
        settings.camera_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
