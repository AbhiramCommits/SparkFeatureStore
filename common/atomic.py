"""Staged writes with atomic promotion for partitioned Parquet tables.

A job writes its output to a sibling staging directory
(``<table>.staging/<run_id>/``), then promotes it partition by partition
into the final table path. Readers of the final path never observe the
staging directory (it is a sibling, not a child), and each partition
directory is either fully present or fully absent -- never half-written.

Consistency notes (see docs/distributed-design.md):

* on POSIX/HDFS, ``rename`` within the same filesystem is atomic;
* a promoted partition that replaces an existing one first moves the old
  partition to ``<table>.trash/``, so a failed promotion can roll back;
* on S3A, ``rename`` is implemented as copy + delete and is NOT atomic --
  readers must gate on the ``_SUCCESS`` marker written after promotion.
"""

from __future__ import annotations

import uuid

from pyspark.sql import DataFrame, SparkSession

from common.config import resolve_path
from common.logging import get_logger

log = get_logger(__name__)

STAGING_SUFFIX = ".staging"
TRASH_SUFFIX = ".trash"
SUCCESS_MARKER = "_SUCCESS"


def staging_path(path: str, run_id: str | None = None) -> str:
    """Sibling staging directory for a table path."""
    resolved = resolve_path(path)
    return f"{resolved}{STAGING_SUFFIX}/{run_id or uuid.uuid4().hex}"


def _filesystem(spark: SparkSession, path: str):
    jvm = spark._jvm
    jpath = jvm.org.apache.hadoop.fs.Path(path)
    return jpath.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration()), jvm


def _clean_stale_staging(spark: SparkSession, staged: str) -> None:
    """Drop older staging run dirs for the same table (keep the current one)."""
    fs, _ = _filesystem(spark, staged)
    root = _parent_path(spark, staged)
    if not fs.exists(root):
        return
    keep = staged.rstrip("/").rsplit("/", 1)[-1]
    for status in fs.listStatus(root):
        if status.isDirectory() and status.getPath().getName() != keep:
            fs.delete(status.getPath(), True)


def _parent_path(spark: SparkSession, path: str):
    jvm = spark._jvm
    return jvm.org.apache.hadoop.fs.Path(path).getParent()


def _promote_one(fs, src, dst) -> None:
    """Rename one partition dir from staging to its final location."""
    if not fs.rename(src, dst):
        raise RuntimeError(f"promote failed: rename {src} -> {dst}")


def _is_partition_dir(name: str) -> bool:
    return not name.startswith(".") and not name.startswith("_")


def _remove_empty_parent(fs, jvm, staged_p) -> None:
    """Remove the staging root once its last run dir is gone."""
    parent = staged_p.getParent()
    if fs.exists(parent):
        try:
            if len(fs.listStatus(parent)) == 0:
                fs.delete(parent, True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not remove empty staging root %s: %s", parent, exc)


def write_parquet_staged(
    spark: SparkSession,
    df: DataFrame,
    path: str,
    partition_by: list[str] | None = None,
    run_id: str | None = None,
    mode: str = "overwrite",
    compression: str = "snappy",
) -> str:
    """Write ``df`` into the staging dir and return its path.

    Nothing under the final table path is touched yet.
    """
    staged = staging_path(path, run_id)
    _clean_stale_staging(spark, staged)
    log.info("Writing staged Parquet: %s (partition_by=%s)", staged, partition_by)
    writer = df.write.mode(mode).option("compression", compression)
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(staged)
    return staged


def promote_staged(spark: SparkSession, staged: str, final: str) -> int:
    """Atomically swap the staged table into the final path.

    The whole table is promoted with a single rename: the existing final
    table (if any) is moved to a trash dir, the staged dir is renamed into
    place, a ``_SUCCESS`` marker is written, and the trash is removed. On
    any failure the old table is restored (best effort), so readers observe
    either the complete old table or the complete new table -- never a mix.
    Returns the number of partition directories in the promoted table.
    """
    fs, jvm = _filesystem(spark, staged)
    staged_p = jvm.org.apache.hadoop.fs.Path(staged)
    final_p = jvm.org.apache.hadoop.fs.Path(final)
    trash_p = jvm.org.apache.hadoop.fs.Path(f"{final}{TRASH_SUFFIX}")
    if not fs.exists(staged_p):
        raise FileNotFoundError(f"staging dir does not exist: {staged}")

    had_existing = fs.exists(final_p)
    partition_count = 1
    try:
        if had_existing:
            if fs.exists(trash_p):
                fs.delete(trash_p, True)
            fs.mkdirs(trash_p)
            if not fs.rename(final_p, jvm.org.apache.hadoop.fs.Path(trash_p, "table")):
                raise RuntimeError(f"promote failed: could not move {final_p} to trash")
        if fs.exists(staged_p):
            partition_count = len(
                [
                    status
                    for status in fs.listStatus(staged_p)
                    if status.isDirectory() and _is_partition_dir(status.getPath().getName())
                ]
            )
        _promote_one(fs, staged_p, final_p)
        # Success marker is written only after the swap is in place.
        marker = jvm.org.apache.hadoop.fs.Path(final_p, SUCCESS_MARKER)
        fs.create(marker, True).close()
        fs.delete(trash_p, True)
        _remove_empty_parent(fs, jvm, staged_p)
        log.info("Promoted table %s -> %s (%d partitions)", staged, final, partition_count)
        return max(partition_count, 1)
    except Exception:
        _rollback_table(fs, jvm, final_p, trash_p, had_existing)
        raise


def _rollback_table(fs, jvm, final_p, trash_p, had_existing: bool) -> None:
    """Restore the old table if it was moved to trash (best effort)."""
    if not had_existing:
        return
    trashed = jvm.org.apache.hadoop.fs.Path(trash_p, "table")
    try:
        if fs.exists(trashed):
            if fs.exists(final_p):
                fs.delete(final_p, True)
            fs.rename(trashed, final_p)
    except Exception as exc:  # noqa: BLE001
        log.error("Rollback after failed promotion was incomplete: %s", exc)


def atomic_write_parquet(
    spark: SparkSession,
    df: DataFrame,
    path: str,
    partition_by: list[str] | None = None,
    run_id: str | None = None,
    mode: str = "overwrite",
    compression: str = "snappy",
) -> int:
    """Write ``df`` to ``path`` via staging + atomic promotion.

    Returns the number of promoted partitions (1 when unpartitioned).
    """
    staged = write_parquet_staged(
        spark,
        df,
        path,
        partition_by=partition_by,
        run_id=run_id,
        mode=mode,
        compression=compression,
    )
    return promote_staged(spark, staged, resolve_path(path))
