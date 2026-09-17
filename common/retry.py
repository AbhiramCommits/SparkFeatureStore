"""Exponential backoff retry for transient failures only.

Retries are bounded, jittered, and classify errors: S3 5xx, connection
resets and timeouts are retried; data-quality failures (e.g.
:class:`DataQualityError`) and other permanent errors are never retried.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from common.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class DataQualityError(Exception):
    """Permanent data-quality failure. Never retried."""


# Substrings matched (case-insensitive) against the exception class name +
# message to classify transient errors. S3/HTTP 5xx, throttling, timeouts
# and connection resets are transient by nature.
TRANSIENT_MARKERS = (
    "connection reset",
    "reset by peer",
    "connection refused",
    "connect exception",
    "timed out",
    "timeout",
    "broken pipe",
    "eof",
    "503",
    "502",
    "504",
    "500 internal server error",
    "service unavailable",
    "amazons3exception",
    "throttl",
    "too many requests",
    "429",
)

_ALWAYS_TRANSIENT = (TimeoutError, ConnectionError, BrokenPipeError)


def is_transient_error(exc: BaseException) -> bool:
    """Return True when ``exc`` looks like a transient infrastructure error."""
    if isinstance(exc, _ALWAYS_TRANSIENT):
        return True
    if isinstance(exc, DataQualityError):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def retry(
    fn: Callable[[], T],
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 1.0,
    is_retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Call ``fn`` with exponential backoff + full jitter.

    Args:
        fn: callable to invoke.
        attempts: total number of tries (>= 1).
        base_delay: initial backoff in seconds.
        max_delay: upper bound on any single backoff in seconds.
        jitter: fraction of jitter applied to the computed delay (full
            jitter == 1.0 picks uniformly in [0, delay]).
        is_retryable: optional predicate; defaults to
            :func:`is_transient_error`.

    Raises:
        The last exception if ``fn`` never succeeds (after ``attempts``
        tries) or if the failure is not retryable.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    predicate = is_retryable or is_transient_error
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not predicate(exc):
                raise
            raw_delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter >= 1.0:
                delay = random.uniform(0.0, raw_delay)  # full jitter
            else:
                spread = raw_delay * jitter / 2.0
                delay = random.uniform(raw_delay - spread, raw_delay + spread)
            log.warning(
                "Retryable failure (attempt %d/%d): %s: %s -- retrying in %.2fs",
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
