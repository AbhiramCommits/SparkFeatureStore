"""Shared training-data loader for both frameworks.

Loads the silver ``trip_features`` table with PyArrow and applies a
TIME-BASED train/val/test split (never random) to match the
point-in-time guarantee: test rows always come from the last days, val
from the days before, and train strictly before that. Both jobs call the
same function, so they see identical frames and row counts by
construction.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common.config import load_config, resolve_path
from common.logging import get_logger

log = get_logger(__name__)

LABEL_COL = "tip_pct"
TS_COL = "pickup_datetime"
DATE_COL = "event_date"

# Features deliberately exclude fare_amount / tip_amount: tip_pct is
# computed FROM those columns, so including them would leak the label.
NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "hour_of_day",
    "pu_pickup_count_1h",
    "pu_pickup_count_6h",
    "pu_pickup_count_24h",
    "pu_mean_fare_1h",
    "pu_mean_fare_6h",
    "pu_mean_fare_24h",
    "driver_trip_count_7d",
    "driver_trip_count_30d",
    "driver_mean_trip_distance_7d",
    "driver_mean_trip_distance_30d",
    "driver_mean_tip_ratio_7d",
    "driver_mean_tip_ratio_30d",
]
CATEGORICAL_FEATURES = ["pu_location_id", "vendor_id"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Time-based split windows for the Jan-2023 dataset (half-open intervals).
SPLIT_WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2023-01-01", "2023-01-20"),
    "val": ("2023-01-20", "2023-01-25"),
    "test": ("2023-01-25", "2023-02-01"),
}


def _read_table(path: str, env: str | None = None, columns: list[str] | None = None) -> pa.Table:
    """Read a parquet table from the local FS or S3/S3A.

    S3 behavior comes entirely from conf/<env>.yaml: with an ``s3a.endpoint``
    configured (local MinIO) an explicit filesystem is built; with no
    endpoint the default AWS S3 filesystem is used and credentials come from
    the pod identity (IRSA) or the standard AWS credential chain -- so only
    the config profile changes between local and AWS.
    """
    resolved = resolve_path(path)
    if not (resolved.startswith("s3://") or resolved.startswith("s3a://")):
        return pq.read_table(resolved, columns=columns)

    cfg = load_config(env)
    s3 = cfg.get("s3a", {}) or {}
    endpoint = s3.get("endpoint") or ""
    bucket_key = resolved.split("://", 1)[1]
    fs_kwargs: dict = {}
    if endpoint:
        scheme = endpoint.split("://", 1)[0] if "://" in endpoint else "http"
        host = endpoint.split("://", 1)[1] if "://" in endpoint else endpoint
        fs_kwargs["endpoint_override"] = host
        fs_kwargs["scheme"] = scheme
        if s3.get("access_key"):
            fs_kwargs["access_key"] = s3["access_key"]
        if s3.get("secret_key"):
            fs_kwargs["secret_key"] = s3["secret_key"]
    fs = pa.fs.S3FileSystem(**fs_kwargs)
    return pq.read_table(bucket_key, filesystem=fs, columns=columns)


def load_training_data(
    path: str,
    split: dict[str, tuple[str, str]] | None = None,
    env: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Return ({split: frame}, {split: row_count}) for the trip_features table.

    Rows are filtered to the split range, junk dates before the first split
    window are dropped, and rows with a null label or categorical key are
    removed (identically for both frameworks).
    """
    split = split or SPLIT_WINDOWS
    # select columns at read time to keep the in-memory frame small
    # (hour_of_day is derived from pickup_datetime, not stored)
    read_columns = [c for c in FEATURE_COLUMNS if c != "hour_of_day"] + [
        LABEL_COL,
        DATE_COL,
        TS_COL,
    ]
    table = _read_table(path, env=env, columns=read_columns)
    df = table.to_pandas()

    first_start = min(lo for lo, _ in split.values())
    df = df[df[DATE_COL].astype(str) >= first_start]
    df = df[df[LABEL_COL].notna()]
    df = df.dropna(subset=CATEGORICAL_FEATURES)
    df = df.copy()
    df["hour_of_day"] = pd.to_datetime(df[TS_COL]).dt.hour.astype("float64")

    frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    for name, (lo, hi) in split.items():
        mask = (df[DATE_COL].astype(str) >= lo) & (df[DATE_COL].astype(str) < hi)
        # DATE_COL is kept as a metadata column (the jobs select features
        # explicitly, so it is never fed to a model).
        frames[name] = df.loc[mask, FEATURE_COLUMNS + [LABEL_COL, DATE_COL]].reset_index(drop=True)
        counts[name] = int(mask.sum())

    total = sum(counts.values())
    assert total == sum(len(f) for f in frames.values()), "split frames are inconsistent"
    assert total > 0, "no training rows found in the split windows"
    log.info(
        "Loaded %d rows: train=%d val=%d test=%d",
        total,
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return frames, counts


def verify_identical_row_counts(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    name_a: str = "framework A",
    name_b: str = "framework B",
) -> None:
    """Assert two jobs saw identical row counts (they share this loader)."""
    assert counts_a == counts_b, f"row counts differ: {name_a}={counts_a} vs {name_b}={counts_b}"
