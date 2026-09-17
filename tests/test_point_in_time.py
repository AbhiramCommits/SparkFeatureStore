"""Point-in-time correctness tests with hand-computed answers.

Every expected value below is computed by hand from the input rows. The
strictness of the frames (end offset -1) and the backward-only as-of join
are what make these tests fail if the implementation leaks same-timestamp
or future rows.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from features import point_in_time as pit
from features.build_silver import add_tip_pct


def _ts_frame(spark: SparkSession, rows, schema: str):
    return spark.createDataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Trailing windows (rangeBetween)
# ---------------------------------------------------------------------------


def test_trailing_window_excludes_same_timestamp_rows(spark: SparkSession) -> None:
    """A leaky window (frame end >= 0) would include same-second rows."""
    df = _ts_frame(
        spark,
        [("A", 100, 10.0), ("A", 200, 20.0), ("A", 200, 5.0), ("A", 500, 30.0)],
        "zone STRING, ts LONG, fare DOUBLE",
    )
    out = pit.add_trailing_windows(
        df,
        keys=["zone"],
        ts_unix_col="ts",
        windows=[("1h", -3600, -1)],
        aggregations=[("cnt", "count", "fare"), ("avg_fare", "avg", "fare")],
    )
    rows = out.orderBy("ts", "fare").collect()

    # ts=100: nothing before it -> 0 / NULL
    assert rows[0].ts == 100
    assert rows[0].cnt_1h == 0
    assert rows[0].avg_fare_1h is None

    # both rows at ts=200 must see ONLY the ts=100 row (no same-timestamp
    # leakage): count 1, mean 10.0 -- a leaky impl (end offset 0) gives 3.
    assert rows[1].cnt_1h == 1 and rows[2].cnt_1h == 1
    assert rows[1].avg_fare_1h == 10.0 and rows[2].avg_fare_1h == 10.0

    # ts=500 sees ts in [500-3600, 499] -> rows at 100, 200, 200
    assert rows[3].cnt_1h == 3
    assert rows[3].avg_fare_1h == pytest.approx(35.0 / 3.0)


def test_trailing_window_excludes_future_rows(spark: SparkSession) -> None:
    """Row at ts=100 must not see the future row at ts=200, even for
    non-negative frame offsets."""
    df = _ts_frame(
        spark,
        [("A", 100, 10.0), ("A", 200, 20.0)],
        "zone STRING, ts LONG, fare DOUBLE",
    )
    out = pit.add_trailing_windows(
        df,
        keys=["zone"],
        ts_unix_col="ts",
        windows=[("1h", -3600, -1)],
        aggregations=[("cnt", "count", "fare")],
    )
    first, second = out.orderBy("ts").collect()
    assert first.cnt_1h == 0
    assert second.cnt_1h == 1  # only the ts=100 row


def test_day_grain_window_excludes_current_day(spark: SparkSession) -> None:
    """Day-grain frame [-7d, -1d]: day D sees days D-7..D-1 only."""
    day = 86400
    df = _ts_frame(
        spark,
        [("V", day, 10.0), ("V", 2 * day, 20.0), ("V", 3 * day, 30.0)],
        "vendor STRING, day_ts LONG, dist DOUBLE",
    )
    out = pit.add_trailing_windows(
        df,
        keys=["vendor"],
        ts_unix_col="day_ts",
        windows=[("7d", -7 * day, -day)],
        aggregations=[("cnt", "count", "dist"), ("avg_d", "avg", "dist")],
    )
    rows = out.orderBy("day_ts").collect()
    assert [r.cnt_7d for r in rows] == [0, 1, 2]
    assert rows[0].avg_d_7d is None
    assert rows[1].avg_d_7d == 10.0
    assert rows[2].avg_d_7d == 15.0


def test_windows_partition_by_keys(spark: SparkSession) -> None:
    df = _ts_frame(
        spark,
        [("A", 100, 1.0), ("B", 100, 10.0), ("A", 200, 2.0)],
        "zone STRING, ts LONG, fare DOUBLE",
    )
    out = pit.add_trailing_windows(
        df,
        keys=["zone"],
        ts_unix_col="ts",
        windows=[("1h", -3600, -1)],
        aggregations=[("cnt", "count", "fare")],
    )
    rows = {r.zone: r.cnt_1h for r in out.collect()}
    assert rows["A"] == 1  # A@200 sees A@100; B's row is in another partition
    assert rows["B"] == 0


# ---------------------------------------------------------------------------
# as_of_join
# ---------------------------------------------------------------------------


def _asof_frames(spark: SparkSession):
    spine = _ts_frame(
        spark,
        [(1, "z", 300), (2, "z", 100)],
        "id INT, k STRING, ts LONG",
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    feat = _ts_frame(
        spark,
        [("z", 200, "at200"), ("z", 300, "at300"), ("z", 400, "at400")],
        "k STRING, ts LONG, v STRING",
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    return spine, feat


def test_as_of_join_never_looks_forward(spark: SparkSession) -> None:
    """Spine@300 must match feat@300, never the future feat@400."""
    spine, feat = _asof_frames(spark)
    out = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=1000, prefix="asof")
    got = {r.id: r.v for r in out.collect()}
    assert got[1] == "at300"  # a look-ahead implementation would return "at400"
    assert got[2] is None  # nothing <= 100 exists


def test_as_of_join_picks_most_recent_backward_match(spark: SparkSession) -> None:
    spine, feat = _asof_frames(spark)
    out = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=1000, prefix="asof")
    row1 = [r for r in out.collect() if r.id == 1][0]
    # both at200 and at300 are <= 300: the join must pick at300
    assert row1.v == "at300"
    assert row1.asof_ts is not None  # matched feature timestamp attached


def test_as_of_join_respects_tolerance(spark: SparkSession) -> None:
    spine = _ts_frame(spark, [(1, "z", 300)], "id INT, k STRING, ts LONG").withColumn(
        "ts", F.col("ts").cast("timestamp")
    )
    feat = _ts_frame(
        spark, [("z", 100, "at100"), ("z", 500, "at500")], "k STRING, ts LONG, v STRING"
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    within = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=100, prefix="asof")
    assert within.collect()[0].v is None  # at100 is 200s away

    outside = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=250, prefix="asof")
    assert outside.collect()[0].v == "at100"


def test_as_of_join_isolates_keys(spark: SparkSession) -> None:
    spine = _ts_frame(
        spark, [(1, "x", 300), (2, "y", 300)], "id INT, k STRING, ts LONG"
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    feat = _ts_frame(
        spark, [("x", 250, "x250"), ("y", 100, "y100")], "k STRING, ts LONG, v STRING"
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    out = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=1000, prefix="asof")
    got = {r.id: r.v for r in out.collect()}
    assert got == {1: "x250", 2: "y100"}


def test_as_of_join_renames_colliding_columns(spark: SparkSession) -> None:
    spine = _ts_frame(
        spark,
        [(1, "z", 300, "2023-01-05")],
        "id INT, k STRING, ts LONG, event_date STRING",
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    feat = _ts_frame(
        spark,
        [("z", 200, "2023-01-04", 7.0)],
        "k STRING, ts LONG, event_date STRING, val DOUBLE",
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    out = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=1000, prefix="asof")
    row = out.collect()[0]
    assert row.event_date == "2023-01-05"  # spine value preserved
    assert row.asof_event_date == "2023-01-04"  # feature value prefixed
    assert row.val == 7.0


def test_as_of_join_zero_tolerance_matches_exact_ts_only(spark: SparkSession) -> None:
    spine = _ts_frame(spark, [(1, "z", 300)], "id INT, k STRING, ts LONG").withColumn(
        "ts", F.col("ts").cast("timestamp")
    )
    feat = _ts_frame(
        spark, [("z", 299, "at299"), ("z", 300, "at300")], "k STRING, ts LONG, v STRING"
    ).withColumn("ts", F.col("ts").cast("timestamp"))
    out = pit.as_of_join(spine, feat, ts_col="ts", keys=["k"], tolerance=0, prefix="asof")
    assert out.collect()[0].v == "at300"  # the previous second must NOT match


# ---------------------------------------------------------------------------
# tip_pct label
# ---------------------------------------------------------------------------


def test_tip_pct_clipped(spark: SparkSession) -> None:
    df = _ts_frame(
        spark,
        [
            (10.0, 2.0),
            (10.0, 20.0),
            (10.0, -5.0),
            (0.0, 5.0),
            (10.0, None),
        ],
        "fare_amount DOUBLE, tip_amount DOUBLE",
    )
    vals = [r.tip_pct for r in add_tip_pct(df).collect()]
    assert vals[0] == pytest.approx(0.2)
    assert vals[1] == 1.0  # clipped at 1
    assert vals[2] == 0.0  # clipped at 0
    assert vals[3] is None  # fare <= 0 -> undefined
    assert vals[4] is None  # tip missing -> undefined
