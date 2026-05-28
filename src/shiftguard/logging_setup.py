"""Structured trace logging -> console + a timestamped run file.

The file handler's output is the project's demo deliverable: a detailed
thought/action/observation trace of an agent run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .config import get_settings

_LOGGER_NAME = "shiftguard"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    run_name: str = "run",
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure the `shiftguard` logger with console + file handlers.

    Returns the configured logger and writes to `<log_dir>/<run_name>_<ts>.log`.
    Safe to call once per run; existing handlers are replaced.
    """
    target_dir = Path(log_dir) if log_dir is not None else get_settings().logs_path
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = target_dir / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("logging initialized -> %s", log_path)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the shared `shiftguard` logger, optionally a named child."""
    base = logging.getLogger(_LOGGER_NAME)
    return base.getChild(name) if name else base
