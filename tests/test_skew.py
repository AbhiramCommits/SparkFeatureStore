"""Tests for hot-key salting (features/skew.py)."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features.skew import SALT_COL, add_salt, detect_hot_keys, salted_aggregate


def _df(spark: SparkSession, rows: list[tuple]):
    return spark.createDataFrame(rows, "k STRING, v DOUBLE")


def test_detect_hot_keys(spark: SparkSession) -> None:
    rows = [("hot", 1.0)] * 12 + [("cold1", 1.0)] * 5 + [("cold2", 1.0)] * 3
    hot = detect_hot_keys(_df(spark, rows), "k", threshold=8)
    assert hot == ["hot"]

    none_hot = detect_hot_keys(_df(spark, rows), "k", threshold=100)
    assert none_hot == []


def test_salted_aggregate_matches_unsalted(spark: SparkSession) -> None:
    rows = [("hot", float(i)) for i in range(100)]
    rows += [("cold1", float(i)) for i in range(5)]
    rows += [("cold2", 42.0)]
    df = _df(spark, rows)

    salted = salted_aggregate(
        df,
        group_cols=["k"],
        hot_col="k",
        hot_keys=["hot"],
        salt_buckets=8,
        aggregates=[("v", "count"), ("v", "avg"), ("v", "sum")],
        seed=42,
    )
    plain = df.groupBy("k").agg(
        F.count(F.lit(1)).alias("v_count"),
        F.avg("v").alias("v_avg"),
        F.sum("v").alias("v_sum"),
    )
    assert SALT_COL not in salted.columns

    plain_map = {r.k: (r.v_count, r.v_sum, r.v_avg) for r in plain.collect()}
    salted_map = {r.k: (r.v_count, r.v_sum, r.v_avg) for r in salted.collect()}
    assert set(plain_map) == set(salted_map)
    for key in plain_map:
        pc, ps, pa = plain_map[key]
        sc, ss, sa = salted_map[key]
        assert pc == sc
        assert ss == pytest.approx(ps, rel=1e-9, abs=1e-6)
        assert (pa is None and sa is None) or abs(pa - sa) < 1e-9


def test_salt_applied_only_to_hot_keys(spark: SparkSession) -> None:
    df = _df(spark, [("hot", 1.0), ("hot", 2.0), ("cold", 3.0)])
    out = add_salt(df, "k", ["hot"], salt_buckets=8, seed=42).collect()
    for row in out:
        if row.k == "hot":
            assert 0 <= row._salt <= 7
        else:
            assert row._salt == 0


def test_salt_bucket_range_is_respected(spark: SparkSession) -> None:
    df = _df(spark, [("hot", float(i)) for i in range(200)])
    out = add_salt(df, "k", ["hot"], salt_buckets=16, seed=None).collect()
    assert {r._salt for r in out} <= set(range(16))


def test_invalid_aggregation_op_rejected(spark: SparkSession) -> None:
    df = _df(spark, [("a", 1.0)])
    with pytest.raises(ValueError):
        salted_aggregate(df, ["k"], "k", [], 4, [("v", "median")])
