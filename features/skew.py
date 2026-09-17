"""Hot-key salting for skewed aggregations.

NYC taxi data is heavily skewed: a handful of PULocationIDs (Midtown,
airports) receive most trips. Grouping by such keys concentrates the whole
aggregation into one or two tasks.

Salting splits each hot key into ``salt_buckets`` shards: partial
aggregates are computed per (key, salt) -- spreading hot keys across many
tasks -- and then combined by a second aggregation over the real keys.
Cold keys are left untouched (salt 0), so their partitions do not multiply.

Hot keys are detected at runtime from an approximate frequency count
(sampled groupBy) above a configurable threshold, so only the actually
skewed keys are salted.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SALT_COL = "_salt"


def detect_hot_keys(
    df: DataFrame,
    key_col: str,
    threshold: int,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> list:
    """Return keys with (estimated) row count >= ``threshold``.

    Uses a Bernoulli sample of the dataframe for an approximate frequency
    count; counts are scaled by ``1 / sample_fraction``.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    sampled = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
    counts = sampled.groupBy(key_col).count()
    estimated = counts.withColumn(
        "est_count", (F.col("count") / F.lit(sample_fraction)).cast("long")
    )
    hot = [row[key_col] for row in estimated.filter(F.col("est_count") >= threshold).collect()]
    return hot


def add_salt(
    df: DataFrame, hot_col: str, hot_keys: list, salt_buckets: int, seed: int | None = None
) -> DataFrame:
    """Append ``_salt`` in [0, salt_buckets) to hot-key rows; 0 otherwise."""
    if salt_buckets < 1:
        raise ValueError("salt_buckets must be >= 1")
    salt = F.when(
        F.col(hot_col).isin(hot_keys) if hot_keys else F.lit(False),
        F.floor(F.rand(seed) * F.lit(salt_buckets)).cast("int"),
    ).otherwise(0)
    return df.withColumn(SALT_COL, salt)


def salted_aggregate(
    df: DataFrame,
    group_cols: list[str],
    hot_col: str,
    hot_keys: list,
    salt_buckets: int,
    aggregates: list[tuple[str, str]],
    seed: int | None = None,
) -> DataFrame:
    """Two-phase salted aggregation that matches the unsalted result exactly.

    Args:
        df: input rows; must contain ``group_cols`` + ``hot_col``.
        group_cols: final grouping columns (includes ``hot_col`` usually).
        hot_col: the skewed column salting is applied to.
        hot_keys: keys to salt (see :func:`detect_hot_keys`).
        salt_buckets: number of shards per hot key.
        aggregates: (col, op) pairs; op in {"count", "sum", "avg"}.
            Output columns are named ``{col}_{op}``.

    Returns:
        A dataframe grouped by ``group_cols`` with one column per aggregate.
    """
    salted = add_salt(df, hot_col, hot_keys, salt_buckets, seed=seed)

    partial_exprs: list = []
    specs: list[tuple[str, int]] = []
    for i, (col, op) in enumerate(aggregates):
        if op == "count":
            partial_exprs.append(F.count(F.lit(1)).alias(f"_c{i}"))
            specs.append(("count", i))
        elif op == "sum":
            partial_exprs.append(F.sum(F.col(col)).alias(f"_s{i}"))
            specs.append(("sum", i))
        elif op == "avg":
            partial_exprs.append(F.sum(F.col(col)).alias(f"_s{i}"))
            partial_exprs.append(F.count(F.col(col)).alias(f"_n{i}"))
            specs.append(("avg", i))
        else:
            raise ValueError(f"unsupported aggregation op: {op}")
    partial = salted.groupBy(*group_cols, SALT_COL).agg(*partial_exprs)

    final_exprs: list = []
    for op, i in specs:
        col, _ = aggregates[i]
        if op == "count":
            final_exprs.append(F.sum(f"_c{i}").alias(f"{col}_count"))
        elif op == "sum":
            final_exprs.append(F.sum(f"_s{i}").alias(f"{col}_sum"))
        else:
            final_exprs.append((F.sum(f"_s{i}") / F.sum(f"_n{i}")).alias(f"{col}_avg"))
    return partial.groupBy(*group_cols).agg(*final_exprs)
