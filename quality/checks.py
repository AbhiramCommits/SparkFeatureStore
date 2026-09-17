"""Data-quality gates that FAIL the job instead of writing bad partitions.

Every gate returns a :class:`CheckResult`; :func:`run_gates` records all
results to the Postgres ``quality_results`` table and raises
:class:`common.retry.DataQualityError` when any gate fails -- which aborts
promotion from staging, so the previous good table stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from common.logging import get_logger
from common.retry import DataQualityError
from quality.contracts import TableContract, check_contract
from quality.registry import record_result

log = get_logger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    measured: str
    threshold: str


def check_null_rates(df: DataFrame, ceilings: dict[str, float]) -> list[CheckResult]:
    """Null fraction per column must stay under the given ceiling."""
    if not ceilings:
        return []
    row = (
        df.agg(*[F.avg(F.col(col).isNull().cast("double")).alias(col) for col in ceilings])
        .first()
        .asDict()
    )
    return [
        CheckResult(
            f"null_rate:{col}",
            row[col] <= ceiling,
            f"{row[col]:.4f}",
            f"<= {ceiling}",
        )
        for col, ceiling in ceilings.items()
    ]


def check_primary_key_unique(df: DataFrame, keys: tuple[str, ...]) -> CheckResult:
    """Duplicate primary-key rows must be zero."""
    dups = df.groupBy(*keys).count().filter(F.col("count") > 1).count()
    return CheckResult(
        "primary_key_unique",
        dups == 0,
        f"{dups} duplicate keys",
        "0 duplicate keys",
    )


def check_row_count_floor(df: DataFrame, min_rows: int) -> CheckResult:
    n = df.count()
    return CheckResult("row_count_floor", n >= min_rows, str(n), f">= {min_rows}")


def check_partition_floor(
    spark: SparkSession,
    df: DataFrame,
    existing_path: str,
    event_date_col: str,
    floor_ratio: float,
) -> CheckResult:
    """Each event_date partition must keep at least ``floor_ratio`` of the
    row count it had in the previous table (skipped when no prior table)."""
    current = df.groupBy(event_date_col).count()
    try:
        prior = spark.read.parquet(existing_path).groupBy(event_date_col).count()
    except Exception:  # noqa: BLE001 -- no prior table to compare against
        return CheckResult(
            "partition_row_floor", True, "no prior table", f">= {floor_ratio} of prior"
        )
    joined = (
        current.alias("c")
        .join(prior.alias("p"), on=event_date_col)
        .filter(F.col("c.count") < F.col("p.count") * F.lit(floor_ratio))
    )
    bad = joined.count()
    return CheckResult(
        "partition_row_floor",
        bad == 0,
        f"{bad} partitions below floor",
        f"count >= {floor_ratio} * prior count",
    )


def check_freshness(df: DataFrame, ts_col: str, lower: str, upper: str) -> CheckResult:
    """max(ts_col) must fall inside the expected window."""
    max_ts = df.agg(F.max(F.col(ts_col))).first()[0]
    if max_ts is None:
        return CheckResult("freshness", False, "no timestamps", f"[{lower}, {upper}]")
    passed = str(max_ts) >= lower and str(max_ts) <= upper
    return CheckResult("freshness", passed, f"max({ts_col})={max_ts}", f"[{lower}, {upper}]")


def run_gates(
    spark: SparkSession,
    conn,
    *,
    feature_group: str,
    df: DataFrame,
    contract: TableContract | None = None,
    run_id: str | None = None,
    existing_path: str | None = None,
    event_date_col: str | None = None,
    min_rows: int | None = None,
    partition_floor_ratio: float | None = None,
    null_rate_ceilings: dict[str, float] | None = None,
    freshness: dict | None = None,
) -> list[CheckResult]:
    """Run all gates, record results, and raise on any failure.

    On violation this raises :class:`DataQualityError` AFTER recording the
    failing check in ``quality_results``; the caller aborts the write and
    the staged output is never promoted.
    """
    results: list[CheckResult] = []
    if contract is not None:
        violations = check_contract(df, contract)
        results.append(
            CheckResult(
                f"contract:{contract.table}",
                not violations,
                f"{len(violations)} violations",
                "0 violations",
            )
        )
        if violations:
            log.error("Contract violations: %s", "; ".join(violations))
    results.extend(check_null_rates(df, null_rate_ceilings or {}))
    if contract is not None and contract.primary_key:
        results.append(check_primary_key_unique(df, contract.primary_key))
    if min_rows is not None:
        results.append(check_row_count_floor(df, min_rows))
    if partition_floor_ratio is not None and existing_path and event_date_col:
        results.append(
            check_partition_floor(spark, df, existing_path, event_date_col, partition_floor_ratio)
        )
    if freshness:
        results.append(
            check_freshness(df, freshness["ts_col"], freshness["lower"], freshness["upper"])
        )

    for result in results:
        log.info(
            "quality gate %s: %s (measured=%s threshold=%s)",
            result.name,
            "PASS" if result.passed else "FAIL",
            result.measured,
            result.threshold,
        )
        if conn is not None:
            record_result(
                conn,
                run_id=run_id,
                feature_group=feature_group,
                check_name=result.name,
                status="passed" if result.passed else "failed",
                measured_value=result.measured,
                threshold=result.threshold,
                checked_at=datetime.now(timezone.utc),
            )

    failures = [r for r in results if not r.passed]
    if failures:
        raise DataQualityError(
            f"quality gates failed for '{feature_group}': "
            + "; ".join(
                f"{r.name} (measured={r.measured}, threshold={r.threshold})" for r in failures
            )
        )
    log.info("All %d quality gates passed for '%s'", len(results), feature_group)
    return results
