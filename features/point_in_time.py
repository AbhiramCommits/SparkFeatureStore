"""Point-in-time-correct feature helpers for the silver layer.

Guarantees
----------
1. Every aggregate attached to a row at time T is computed over records with
   ``event_ts < T`` only. Trailing frames end at ``T - 1`` second (event
   grain) or ``T - 1`` day (day grain), so same-timestamp rows never leak.
2. Aggregates are built with Spark window functions over ``rangeBetween`` on
   a unix-seconds column -- never a naive groupBy over the whole dataset.
3. :func:`as_of_join` is a backward as-of join: it matches the most recent
   feature row at or before the spine timestamp (within a tolerance) and
   never looks forward.

Frame offsets are expressed in seconds relative to the current row's
timestamp (negative = past).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window, WindowSpec
from pyspark.sql import functions as F

# "count" counts rows inside the trailing frame (col is ignored).
AGG_FUNCTIONS = {
    "count": lambda _col: F.count(F.lit(1)),
    "avg": lambda col: F.avg(F.col(col)),
}


def trailing_window(
    keys: list[str], ts_unix_col: str, start_offset: int, end_offset: int
) -> WindowSpec:
    """Range window over ``keys`` ordered by ``ts_unix_col`` (unix seconds).

    The frame is ``[current_ts + start_offset, current_ts + end_offset]``.
    Use ``end_offset=-1`` for event grain (excludes the current second) and
    ``end_offset=-86400`` for day grain (excludes the current day).
    """
    return (
        Window.partitionBy(*keys)
        .orderBy(F.col(ts_unix_col).cast("long"))
        .rangeBetween(start_offset, end_offset)
    )


def add_trailing_windows(
    df: DataFrame,
    keys: list[str],
    ts_unix_col: str,
    windows: list[tuple[str, int, int]],
    aggregations: list[tuple[str, str, str]],
) -> DataFrame:
    """Attach trailing aggregates to ``df``.

    Args:
        df: rows to enrich; must contain ``keys`` and ``ts_unix_col``.
        keys: entity key columns the windows partition on.
        ts_unix_col: unix-seconds column used for ordering + range frames.
        windows: (name, start_offset, end_offset) tuples in seconds.
        aggregations: (prefix, fn, col) tuples, fn in {"count", "avg"}.

    Returns:
        ``df`` plus one column per combination, named ``{prefix}_{name}``.
    """
    out = df
    for prefix, fn, col in aggregations:
        agg = AGG_FUNCTIONS[fn](col)
        for name, start_offset, end_offset in windows:
            out = out.withColumn(
                f"{prefix}_{name}",
                agg.over(trailing_window(keys, ts_unix_col, start_offset, end_offset)),
            )
    return out


def as_of_join(
    spine: DataFrame,
    feature_df: DataFrame,
    ts_col: str,
    keys: list[str],
    tolerance: int,
    prefix: str = "asof",
) -> DataFrame:
    """Backward as-of join; never looks forward.

    For every spine row at time T, attaches the feature row with the largest
    ``feature_ts`` such that ``feature_ts <= T`` and ``T - feature_ts <=
    tolerance`` (seconds). Feature rows with ``feature_ts > T`` are never
    matched, so the result can only depend on past data.

    Args:
        spine: rows to enrich; must contain ``keys`` and ``ts_col``.
        feature_df: feature values; must contain ``keys`` and ``ts_col``.
            Must be unique on (keys, ts_col); duplicates are dropped
            (arbitrary row kept).
        ts_col: timestamp column name (must exist in both frames).
        keys: entity key columns to join on.
        tolerance: maximum seconds between T and the matched feature ts.
        prefix: prefix for feature columns that collide with spine columns
            and for the matched feature timestamp column.

    Returns:
        ``spine`` plus the feature columns. Feature columns that collide with
        spine columns are renamed to ``{prefix}_{col}``; the matched feature
        timestamp is attached as ``{prefix}_{ts_col}``. Spine rows without a
        match within tolerance get NULLs for all feature columns.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    for c in keys + [ts_col]:
        if c not in spine.columns:
            raise ValueError(f"spine is missing column: {c}")
        if c not in feature_df.columns:
            raise ValueError(f"feature_df is missing column: {c}")

    spine_id = f"_{prefix}_spine_id"
    feat_ts = f"_{prefix}_feat_ts"
    bucket_size = tolerance + 1

    feat = feature_df.dropDuplicates(keys + [ts_col]).withColumnRenamed(ts_col, feat_ts)
    feat = feat.withColumn(
        "_feat_bucket", (F.unix_timestamp(F.col(feat_ts)) / F.lit(bucket_size)).cast("long")
    )

    spine = spine.withColumn(spine_id, F.monotonically_increasing_id())
    spine = spine.withColumn(
        "_spine_bucket", (F.unix_timestamp(F.col(ts_col)) / F.lit(bucket_size)).cast("long")
    )
    # A feature row within tolerance of T lies in T's bucket or the previous
    # one (bucket size = tolerance + 1), so we probe both buckets with an
    # equi-join and enforce the exact backward condition afterwards. This
    # keeps the join a hash join instead of a full cartesian per key.
    spine_expanded = (
        spine.withColumn("_bucket_off", F.explode(F.array(F.lit(0), F.lit(-1))))
        .withColumn("_join_bucket", F.col("_spine_bucket") + F.col("_bucket_off"))
        .drop("_bucket_off", "_spine_bucket")
    )
    feat_probe = feat.withColumn("_join_bucket", F.col("_feat_bucket")).drop("_feat_bucket")
    spine_cols = set(spine.columns)
    feature_value_cols = [c for c in feat_probe.columns if c not in keys + ["_join_bucket"]]
    # Rename colliding feature columns before the join so the result never
    # contains ambiguous column names.
    collisions = {c: f"{prefix}_{c}" for c in feature_value_cols if c in spine_cols}
    feat_probe = feat_probe.withColumnsRenamed(collisions)
    feature_value_cols = [collisions.get(c, c) for c in feature_value_cols]

    # NOTE: everything below is a single lineage path so the spine (and its
    # spine_id column) is evaluated exactly once. A second evaluation would
    # recompute monotonically_increasing_id with a different partition
    # layout and silently pair the wrong feature rows.
    valid = (F.col(feat_ts) <= F.col(ts_col)) & (
        (F.unix_timestamp(F.col(ts_col)) - F.unix_timestamp(F.col(feat_ts))) <= F.lit(tolerance)
    )
    candidates = spine_expanded.join(feat_probe, on=keys + ["_join_bucket"], how="left")
    ranked = candidates.withColumn(
        "_rn",
        F.row_number().over(
            Window.partitionBy(spine_id).orderBy(F.when(valid, F.col(feat_ts)).desc_nulls_last())
        ),
    )
    result = ranked.filter(F.col("_rn") == 1).drop("_rn", "_join_bucket")
    # Spine rows without a valid match keep a placeholder candidate row;
    # null out its feature values so no future data can leak through.
    for c in feature_value_cols + [feat_ts]:
        result = result.withColumn(c, F.when(valid, F.col(c)))
    return result.withColumnRenamed(feat_ts, f"{prefix}_{ts_col}")
