"""Tests for staged writes + atomic promotion (common/atomic.py).

Includes the kill-mid-write scenario: an injected failure during promotion
must never leave a half-written partition visible, and a re-run must
produce a consistent table without duplicates or leftovers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from common import atomic


def _df(spark: SparkSession, n: int = 6):
    rows = [(i, f"2023-01-0{(i % 3) + 1}") for i in range(1, n + 1)]
    return spark.createDataFrame(rows, "id INT, event_date STRING")


def _partition_dirs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("event_date="))


def test_atomic_write_promotes_partitions(spark: SparkSession, tmp_path: Path) -> None:
    out = tmp_path / "tbl"
    moved = atomic.atomic_write_parquet(
        spark, _df(spark), str(out), partition_by=["event_date"], run_id="r1"
    )
    assert moved == 3
    assert _partition_dirs(out) == [
        "event_date=2023-01-01",
        "event_date=2023-01-02",
        "event_date=2023-01-03",
    ]
    assert (out / "_SUCCESS").exists()
    assert spark.read.parquet(str(out)).count() == 6
    assert not (tmp_path / "tbl.staging").exists()
    assert not (tmp_path / "tbl.trash").exists()


def test_staging_invisible_until_promoted(spark: SparkSession, tmp_path: Path) -> None:
    out = tmp_path / "tbl"
    staged = atomic.write_parquet_staged(
        spark, _df(spark), str(out), partition_by=["event_date"], run_id="r1"
    )
    assert Path(staged).exists()
    assert not out.exists()  # final table untouched while staging
    atomic.promote_staged(spark, staged, str(out))
    assert spark.read.parquet(str(out)).count() == 6


def test_atomic_overwrite_replaces_existing_partitions(spark: SparkSession, tmp_path: Path) -> None:
    out = tmp_path / "tbl"
    atomic.atomic_write_parquet(
        spark, _df(spark, 6), str(out), partition_by=["event_date"], run_id="r1"
    )

    df2 = spark.createDataFrame(
        [(1, "2023-01-01"), (2, "2023-01-01"), (3, "2023-01-02")], "id INT, event_date STRING"
    )
    atomic.atomic_write_parquet(spark, df2, str(out), partition_by=["event_date"], run_id="r2")
    assert spark.read.parquet(str(out)).count() == 3
    assert _partition_dirs(out) == ["event_date=2023-01-01", "event_date=2023-01-02"]


def test_killed_mid_promotion_then_rerun_is_consistent(
    spark: SparkSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected exception mid-promotion: rollback + re-run must leave the
    final table complete, with no duplicate or partial partitions."""
    out = tmp_path / "tbl"
    real_promote_one = atomic._promote_one
    calls = {"n": 0}

    def flaky_promote(fs, src, dst) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected failure mid-promotion")
        return real_promote_one(fs, src, dst)

    monkeypatch.setattr(atomic, "_promote_one", flaky_promote)
    with pytest.raises(RuntimeError):
        atomic.atomic_write_parquet(
            spark, _df(spark), str(out), partition_by=["event_date"], run_id="r1"
        )
    # Rollback: nothing promoted may stay visible; no success marker.
    assert _partition_dirs(out) == []
    assert not (out / "_SUCCESS").exists()

    # Re-run succeeds and leaves a consistent table.
    monkeypatch.setattr(atomic, "_promote_one", real_promote_one)
    atomic.atomic_write_parquet(
        spark, _df(spark), str(out), partition_by=["event_date"], run_id="r2"
    )
    assert spark.read.parquet(str(out)).count() == 6
    dirs = _partition_dirs(out)
    assert len(dirs) == 3
    assert len(set(dirs)) == 3  # no duplicate partitions
    assert not (tmp_path / "tbl.staging").exists()  # stale staging cleaned
    assert not (tmp_path / "tbl.trash").exists()
