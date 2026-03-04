"""Structured logging utilities for PAC-Index experiments."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    structured: bool = False,
) -> logging.Logger:
    """Configure logging for PAC-Index experiments.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
        structured: If True, use JSON structured logging.

    Returns:
        Configured root logger.
    """
    root_logger = logging.getLogger("pac_index")
    root_logger.setLevel(getattr(logging, level.upper()))

    if structured:
        formatter = logging.Formatter(
            json.dumps({
                "timestamp": "%(asctime)s",
                "level": "%(levelname)s",
                "module": "%(module)s",
                "message": "%(message)s",
            })
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class ExperimentTimer:
    """Context manager for timing experiment phases."""

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        self.name = name
        self.logger = logger or logging.getLogger("pac_index")
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "ExperimentTimer":
        self.start_time = time.perf_counter()
        self.logger.info("Starting: %s", self.name)
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        self.logger.info("Completed: %s (%.3f seconds)", self.name, self.elapsed)
