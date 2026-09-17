"""Parquet I/O helpers.

Reads and writes Parquet against either the local filesystem or S3A
(MinIO locally, AWS S3 in production). Paths starting with a URL scheme
(s3a://...) are passed through unchanged; local paths are resolved against
the repo root.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from common.config import resolve_path
from common.logging import get_logger

log = get_logger(__name__)


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    resolved = resolve_path(path)
    log.info("Reading Parquet: %s", resolved)
    return spark.read.parquet(resolved)


def write_parquet(
    df: DataFrame,
    path: str | Path,
    partition_by: list[str] | None = None,
    mode: str = "overwrite",
    compression: str = "snappy",
) -> None:
    resolved = resolve_path(path)
    log.info(
        "Writing Parquet: %s (partition_by=%s, mode=%s, compression=%s)",
        resolved,
        partition_by,
        mode,
        compression,
    )
    writer = df.write.mode(mode).option("compression", compression)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(resolved)


def table_path(base: str | Path, *parts: str) -> str:
    """Join a base path with table parts, e.g. table_path("data/bronze", "trips")."""
    return str(Path(base).joinpath(*parts))
