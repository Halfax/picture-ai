from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "picture_ai"


def configure_logging(log_path: Path, *, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared PictureAI logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.debug("PictureAI logging initialized at %s", log_path)
    return logger


def get_logger() -> logging.Logger:
    """Return the shared PictureAI logger if already configured."""
    return logging.getLogger(_LOGGER_NAME)
