"""Tests for common.retry."""

from __future__ import annotations

import pytest

from common import retry as retry_mod
from common.retry import DataQualityError, is_transient_error, retry


class FakeConnectionReset(ConnectionResetError):
    pass


def test_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionResetError("connection reset by peer")
        return 42

    assert retry(flaky, attempts=5, base_delay=0.01, max_delay=0.1) == 42
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_no_retry_on_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def bad() -> None:
        calls["n"] += 1
        raise ValueError("schema mismatch: expected int, got string")

    with pytest.raises(ValueError):
        retry(bad, attempts=5, base_delay=0.01)
    assert calls["n"] == 1
    assert sleeps == []


def test_no_retry_on_data_quality_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def bad() -> None:
        calls["n"] += 1
        raise DataQualityError("null ratio exceeded threshold")

    with pytest.raises(DataQualityError):
        retry(bad, attempts=5, base_delay=0.01)
    assert calls["n"] == 1
    assert sleeps == []


def test_exhausts_attempts_and_raises_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        retry(flaky, attempts=4, base_delay=0.01, max_delay=0.05)
    assert calls["n"] == 4
    assert len(sleeps) == 3  # no sleep after the last attempt


def test_backoff_bounded_and_jittered(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)

    def flaky() -> None:
        raise ConnectionResetError()

    with pytest.raises(ConnectionResetError):
        retry(flaky, attempts=6, base_delay=1.0, max_delay=8.0, jitter=0.0)
    # exact exponential without jitter: 1, 2, 4, 8, 8 (capped)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_custom_predicate() -> None:
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("retry me")
        return 7

    assert retry(flaky, attempts=5, base_delay=0.01, is_retryable=lambda _e: True) == 7
    assert calls["n"] == 3


def test_transient_error_classification() -> None:
    assert is_transient_error(ConnectionResetError("Connection reset by peer"))
    assert is_transient_error(TimeoutError("connection timed out"))
    assert is_transient_error(BrokenPipeError())
    assert is_transient_error(
        RuntimeError(
            "com.amazonaws.services.s3.model.AmazonS3Exception: "
            "Service Unavailable (Service: Amazon S3; Status Code: 503; ...)"
        )
    )
    assert is_transient_error(RuntimeError("AmazonS3Exception: 500 Internal Server Error"))
    assert not is_transient_error(ValueError("schema mismatch: column missing"))
    assert not is_transient_error(DataQualityError("null ratio too high"))
    assert not is_transient_error(RuntimeError("java.lang.IllegalStateException: bad state"))
