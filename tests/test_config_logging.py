"""Tests for centralized configuration and application logging."""

import logging

from evaluator_optimizer.config import (
    DEFAULT_MAX_OPTIMIZATION_ITERATIONS,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_MEMORY_MAX_AGE_DAYS,
    DEFAULT_MEMORY_MAX_RESULTS,
    DEFAULT_PASS_THRESHOLD,
    LOG_FILE,
    METRIC_PASS_THRESHOLDS,
)
from evaluator_optimizer.logging_config import configure_logging, get_logger


def test_core_runtime_constants_are_centralized() -> None:
    """Core runtime defaults should come from the central configuration."""
    assert DEFAULT_MAX_OPTIMIZATION_ITERATIONS == 2
    assert DEFAULT_MEMORY_CACHE_SIZE == 20
    assert DEFAULT_MEMORY_MAX_RESULTS == 5
    assert DEFAULT_MEMORY_MAX_AGE_DAYS == 365
    assert DEFAULT_PASS_THRESHOLD == 0.80
    assert METRIC_PASS_THRESHOLDS["relevance"] == DEFAULT_PASS_THRESHOLD


def test_logging_configuration_is_idempotent() -> None:
    """Repeated configuration must not add duplicate handlers."""
    logger = configure_logging()
    handler_count = len(logger.handlers)

    configured_again = configure_logging()

    assert configured_again is logger
    assert len(configured_again.handlers) == handler_count
    assert LOG_FILE.parent.exists()


def test_component_logger_uses_package_logger_hierarchy() -> None:
    """Component loggers should be children of the application logger."""
    logger = get_logger("test")

    assert logger.name == "evaluator_optimizer.test"
    assert logger.level == logging.NOTSET
