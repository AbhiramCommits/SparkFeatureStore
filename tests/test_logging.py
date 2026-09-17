"""Tests for common/logging (no Spark required)."""

from __future__ import annotations

import logging

from common import logging as logging_mod


def test_get_logger_returns_logger_named_for_module() -> None:
    logger = logging_mod.get_logger("x.y.z")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "x.y.z"


def test_setup_logging_is_idempotent() -> None:
    logging_mod.setup_logging()
    handlers_before = len(logging.getLogger().handlers)
    logging_mod.setup_logging()
    assert len(logging.getLogger().handlers) == handlers_before
    assert handlers_before >= 1
