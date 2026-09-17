"""Tests for the shared training-data loader (no Spark required)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training import data
from training.data import verify_identical_row_counts


def _write_trip_features(path: Path, n: int = 400, seed: int = 7) -> None:
    import random

    import pandas as pd

    rng = random.Random(seed)
    start = dt.date(2023, 1, 1)
    rows = []
    for i in range(n):
        day = start + dt.timedelta(days=i % 30)
        rows.append(
            {
                "event_date": day.isoformat(),
                "pickup_datetime": str(dt.datetime.combine(day, dt.time(i % 24))),
                "passenger_count": i % 6,
                "trip_distance": round(rng.uniform(0, 20), 2),
                "pu_pickup_count_1h": i % 100,
                "pu_pickup_count_6h": (i * 7) % 400,
                "pu_pickup_count_24h": (i * 13) % 1000,
                "pu_mean_fare_1h": rng.uniform(5, 40),
                "pu_mean_fare_6h": rng.uniform(5, 40),
                "pu_mean_fare_24h": rng.uniform(5, 40),
                "driver_trip_count_7d": (i * 3) % 500,
                "driver_trip_count_30d": (i * 5) % 2000,
                "driver_mean_trip_distance_7d": rng.uniform(0, 20),
                "driver_mean_trip_distance_30d": rng.uniform(0, 20),
                "driver_mean_tip_ratio_7d": rng.uniform(0, 0.3),
                "driver_mean_tip_ratio_30d": rng.uniform(0, 0.3),
                "pu_location_id": (i % 40) + 1,
                "vendor_id": (i % 2) + 1,
                "tip_pct": round(rng.uniform(0, 0.3), 4),
            }
        )
    df = pd.DataFrame(rows)
    # some null labels + null driver features, like the real table
    df.loc[df.index % 37 == 0, "tip_pct"] = None
    df.loc[df.index % 53 == 0, "driver_trip_count_7d"] = None
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def trip_table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("silver") / "trip_features"
    _write_trip_features(path)
    return path


def test_time_split_is_ordered_and_non_overlapping(trip_table: Path) -> None:
    frames, _counts = data.load_training_data(str(trip_table))
    splits = [frames["train"], frames["val"], frames["test"]]
    assert all(len(f) > 0 for f in splits)
    # val dates strictly after train dates, test strictly after val
    for earlier, later in zip(splits, splits[1:]):
        assert earlier["event_date"].max() <= later["event_date"].min()
    # test window is the last days of the range
    assert frames["test"]["event_date"].min() >= "2023-01-25"


def test_row_counts_deterministic_and_consistent(trip_table: Path) -> None:
    _frames_a, counts_a = data.load_training_data(str(trip_table))
    _frames_b, counts_b = data.load_training_data(str(trip_table))
    assert counts_a == counts_b
    verify_identical_row_counts(counts_a, counts_b, name_a="sklearn", name_b="torch")
    assert sum(counts_a.values()) > 0


def test_verify_identical_row_counts_detects_difference() -> None:
    with pytest.raises(AssertionError):
        verify_identical_row_counts({"train": 1}, {"train": 2})


def test_label_leak_columns_are_excluded(trip_table: Path) -> None:
    frames, _counts = data.load_training_data(str(trip_table))
    for frame in frames.values():
        assert data.LABEL_COL in frame.columns
        assert "fare_amount" not in frame.columns
        assert "tip_amount" not in frame.columns
        assert set(data.FEATURE_COLUMNS) <= set(frame.columns)


def test_null_labels_dropped(trip_table: Path) -> None:
    table = pq.read_table(str(trip_table)).to_pandas()
    null_labels = int(table[data.LABEL_COL].isna().sum())
    _frames, counts = data.load_training_data(str(trip_table))
    # synthetic data is all within the split window: kept == rows - null labels
    assert sum(counts.values()) == len(table) - null_labels
    for frame in _frames.values():
        assert frame[data.LABEL_COL].notna().all()


def test_s3a_path_builds_filesystem_from_conf(monkeypatch: pytest.MonkeyPatch) -> None:
    """The S3 branch of _read_table is driven entirely by conf/<env>.yaml."""
    import pyarrow as pa

    fs_kwargs: dict = {}

    def fake_s3fs(**kwargs) -> object:
        fs_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(pa.fs, "S3FileSystem", fake_s3fs)
    monkeypatch.setattr(
        pq,
        "read_table",
        lambda key, filesystem, columns: pa.table({"a": [1]}),
    )
    table = data._read_table("s3a://datalake/silver/trip_features", env="docker", columns=["a"])
    assert table.num_rows == 1
    assert fs_kwargs["endpoint_override"] == "minio:9000"
    assert fs_kwargs["scheme"] == "http"
    assert fs_kwargs["access_key"] == "minioadmin"
    assert fs_kwargs["secret_key"] == "minioadmin"


def test_local_path_does_not_build_s3_filesystem(
    trip_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    def boom(*_args, **_kwargs) -> object:
        raise AssertionError("S3FileSystem must not be used for local paths")

    monkeypatch.setattr(pa.fs, "S3FileSystem", boom)
    table = data._read_table(str(trip_table), columns=["tip_pct"])
    assert table.num_rows > 0
