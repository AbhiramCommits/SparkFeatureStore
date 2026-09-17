"""Quality-gate tests.

Includes the required scenario: a deliberately corrupted input aborts the
run (DataQualityError), records a failure row, and leaves the previous
good partition untouched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from common import atomic
from common.retry import DataQualityError
from quality import checks
from quality.contracts import (
    CONTRACTS,
    ColumnSpec,
    TableContract,
    check_contract,
    enforce_contract,
)


def _df(spark: SparkSession, rows: list[tuple], schema: str):
    return spark.createDataFrame(rows, schema=schema)


FARE_CONTRACT = TableContract(
    table="test.trips",
    columns=(
        ColumnSpec("id", "long", nullable=False),
        ColumnSpec("fare_amount", "double", min_value=-10, max_value=100),
        ColumnSpec("tip_pct", "double", min_value=0, max_value=1),
        ColumnSpec("event_date", "string", nullable=False),
    ),
    primary_key=("id",),
)


def _good_df(spark: SparkSession):
    return _df(
        spark,
        [(1, 10.0, 0.2, "2023-01-01"), (2, 20.0, 0.1, "2023-01-02")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )


def test_contract_type_mismatch_detected(spark: SparkSession) -> None:
    df = _df(
        spark,
        [(1, 10.0, 0.2, "d")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    bad = df.withColumn("fare_amount", F.col("fare_amount").cast("string"))
    violations = check_contract(bad, FARE_CONTRACT)
    assert any("fare_amount" in v for v in violations)


def test_contract_missing_column_detected(spark: SparkSession) -> None:
    df = _df(spark, [(1, 10.0, 0.2)], "id LONG, fare_amount DOUBLE, tip_pct DOUBLE")
    violations = check_contract(df, FARE_CONTRACT)
    assert any("missing column 'event_date'" in v for v in violations)


def test_contract_range_violation_detected(spark: SparkSession) -> None:
    df = _df(
        spark,
        [(1, -50.0, 0.2, "d")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    violations = check_contract(df, FARE_CONTRACT)
    assert any("fare_amount" in v for v in violations)


def test_contract_nullability_violation_detected(spark: SparkSession) -> None:
    df = _df(
        spark,
        [(None, 10.0, 0.2, "d")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    violations = check_contract(df, FARE_CONTRACT)
    assert any("id" in v and "null" in v for v in violations)


def test_enforce_contract_raises_data_quality_error(spark: SparkSession) -> None:
    df = _df(
        spark,
        [(1, -50.0, 0.2, "d")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    with pytest.raises(DataQualityError):
        enforce_contract(df, FARE_CONTRACT)


def test_null_rate_check(spark: SparkSession) -> None:
    df = _df(spark, [(1.0, None), (2.0, None), (3.0, 0.5)], "v DOUBLE, w DOUBLE")
    results = checks.check_null_rates(df, {"w": 0.1})
    assert not results[0].passed
    results = checks.check_null_rates(df, {"w": 0.8})
    assert results[0].passed


def test_primary_key_duplicates_detected(spark: SparkSession) -> None:
    df = _df(spark, [(1,), (1,), (2,)], "id LONG")
    assert not checks.check_primary_key_unique(df, ("id",)).passed
    df = _df(spark, [(1,), (2,)], "id LONG")
    assert checks.check_primary_key_unique(df, ("id",)).passed


def test_row_count_floor(spark: SparkSession) -> None:
    df = _df(spark, [(1,), (2,), (3,)], "id LONG")
    assert checks.check_row_count_floor(df, 3).passed
    assert not checks.check_row_count_floor(df, 10).passed


def test_freshness(spark: SparkSession) -> None:
    df = _df(spark, [("2023-01-05 10:00:00",)], "ts STRING").withColumn(
        "ts", F.col("ts").cast("timestamp")
    )
    ok = checks.check_freshness(df, "ts", "2023-01-01T00:00:00", "2023-02-01T00:00:00")
    assert ok.passed
    stale = checks.check_freshness(df, "ts", "2024-01-01T00:00:00", "2024-02-01T00:00:00")
    assert not stale.passed


def test_run_gates_records_failure_and_raises(spark: SparkSession) -> None:
    conn = MagicMock()
    bad = _df(
        spark,
        [(1, -50.0, 0.2, "2023-01-01"), (2, 20.0, 0.1, "2023-01-02")],
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    with pytest.raises(DataQualityError):
        checks.run_gates(
            spark,
            conn,
            feature_group="test.trips",
            run_id="run-1",
            df=bad,
            contract=FARE_CONTRACT,
            min_rows=1,
        )
    # a failed result row was recorded in quality_results
    cur = conn.cursor.return_value.__enter__.return_value
    inserted = [
        call.args[1]
        for call in cur.execute.call_args_list
        if "quality_results" in str(call.args[0])
    ]
    assert inserted, "no quality_results rows recorded"
    statuses = [params[3] for params in inserted]
    assert "failed" in statuses


def test_corrupted_input_aborts_and_previous_good_partition_untouched(
    spark: SparkSession, tmp_path: Path
) -> None:
    """The required end-to-end scenario.

    A good table is written first; a corrupted batch then fails the gates,
    so the promotion never happens and the previous good table is exactly
    as it was.
    """
    out = tmp_path / "tbl"
    good = _good_df(spark)
    atomic.atomic_write_parquet(
        spark, good, str(out), partition_by=["event_date"], run_id="good-run"
    )
    before = spark.read.parquet(str(out)).orderBy("id").collect()

    corrupted = _df(
        spark,
        [(1, 10.0, 0.2, "2023-01-01"), (3, 5.0, 7.5, "2023-01-03")],  # tip_pct=7.5 out of [0,1]
        "id LONG, fare_amount DOUBLE, tip_pct DOUBLE, event_date STRING",
    )
    conn = MagicMock()
    with pytest.raises(DataQualityError):
        checks.run_gates(
            spark,
            conn,
            feature_group="test.trips",
            run_id="bad-run",
            df=corrupted,
            contract=FARE_CONTRACT,
            min_rows=1,
            existing_path=str(out),
            event_date_col="event_date",
            partition_floor_ratio=0.9,
        )

    # previous good table untouched: same rows, no staging/trash leftovers
    after = spark.read.parquet(str(out)).orderBy("id").collect()
    assert after == before
    assert len(after) == 2
    assert (out / "_SUCCESS").exists()
    assert not (tmp_path / "tbl.staging").exists()
    assert not (tmp_path / "tbl.trash").exists()


def test_run_gates_passes_on_good_data(spark: SparkSession) -> None:
    conn = MagicMock()
    results = checks.run_gates(
        spark,
        conn,
        feature_group="test.trips",
        df=_good_df(spark),
        contract=FARE_CONTRACT,
        min_rows=2,
    )
    assert all(r.passed for r in results)


def test_real_contracts_exist_for_pipeline_tables() -> None:
    for table in (
        "bronze.trips",
        "silver.pickup_zone_demand",
        "silver.driver_trip_history",
        "silver.trip_features",
    ):
        assert table in CONTRACTS
