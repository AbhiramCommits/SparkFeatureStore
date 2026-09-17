"""Tests for scripts.fetch_data (no network access required)."""

from __future__ import annotations

from scripts.fetch_data import month_range


def test_full_year_range() -> None:
    months = month_range("2023-01", "2023-12")
    assert len(months) == 12
    assert months[0] == (2023, 1)
    assert months[-1] == (2023, 12)


def test_limit() -> None:
    assert month_range("2023-01", "2023-12", limit=1) == [(2023, 1)]
    assert month_range("2023-01", "2023-12", limit=3) == [(2023, 1), (2023, 2), (2023, 3)]


def test_cross_year_range() -> None:
    assert month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_single_month() -> None:
    assert month_range("2023-06", "2023-06") == [(2023, 6)]
