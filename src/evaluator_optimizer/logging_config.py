"""Central logging configuration for the Evaluator-Optimizer workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from evaluator_optimizer.config import (
    DEFAULT_ENCODING,
    LOG_DATE_FORMAT,
    LOG_DIRECTORY,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOGGER_NAME,
)


def configure_logging() -> logging.Logger:
    """Configure and return the package logger.

    Configuration is idempotent so repeated workflow construction does not
    create duplicate file or console handlers.
    """
    Path(LOG_DIRECTORY).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding=DEFAULT_ENCODING,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(component: str) -> logging.Logger:
    """Return a configured child logger for a package component."""
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
