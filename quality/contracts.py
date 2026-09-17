"""Declarative schema contracts for pipeline tables.

Each contract pins column names, Spark types, nullability and allowed
value ranges. Contracts are enforced BEFORE any write: a violation raises
:class:`common.retry.DataQualityError` and the staged output is never
promoted.

Range bounds are wide enough for legitimate TLC adjustments (e.g. negative
fare corrections exist in the raw data) but tight enough to catch gross
corruption.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)

from common.logging import get_logger
from common.retry import DataQualityError

log = get_logger(__name__)

SPARK_TYPES = {
    "boolean": BooleanType(),
    "date": DateType(),
    "double": DoubleType(),
    "int": IntegerType(),
    "long": LongType(),
    "string": StringType(),
    "timestamp": TimestampType(),
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class TableContract:
    table: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()


CONTRACTS: dict[str, TableContract] = {
    "bronze.trips": TableContract(
        table="bronze.trips",
        columns=(
            ColumnSpec("vendor_id", "long"),
            ColumnSpec("pickup_datetime", "timestamp", nullable=False),
            ColumnSpec("dropoff_datetime", "timestamp"),
            ColumnSpec("passenger_count", "long", min_value=0, max_value=9),
            ColumnSpec("trip_distance", "double", min_value=0, max_value=5000),
            ColumnSpec("rate_code_id", "long", min_value=1, max_value=99),
            ColumnSpec("store_and_fwd_flag", "string"),
            ColumnSpec("pu_location_id", "long", min_value=1, max_value=265),
            ColumnSpec("do_location_id", "long", min_value=1, max_value=265),
            ColumnSpec("payment_type", "long", min_value=0, max_value=6),
            ColumnSpec("fare_amount", "double", min_value=-1000, max_value=5000),
            ColumnSpec("extra", "double", min_value=-10, max_value=100),
            ColumnSpec("mta_tax", "double", min_value=-10, max_value=100),
            ColumnSpec("tip_amount", "double", min_value=-200, max_value=5000),
            ColumnSpec("tolls_amount", "double", min_value=-100, max_value=1000),
            ColumnSpec("improvement_surcharge", "double", min_value=-10, max_value=10),
            ColumnSpec("total_amount", "double", min_value=-1000, max_value=10000),
            ColumnSpec("congestion_surcharge", "double", min_value=-10, max_value=10),
            ColumnSpec("airport_fee", "double", min_value=-10, max_value=10),
            ColumnSpec("ingest_ts", "timestamp", nullable=False),
            ColumnSpec("event_date", "date", nullable=False),
        ),
    ),
    "silver.pickup_zone_demand": TableContract(
        table="silver.pickup_zone_demand",
        columns=(
            ColumnSpec("pu_location_id", "long", nullable=False, min_value=1, max_value=265),
            ColumnSpec("pickup_datetime", "timestamp", nullable=False),
            ColumnSpec("event_date", "date", nullable=False),
            ColumnSpec(
                "pu_pickup_count_1h", "long", nullable=False, min_value=0, max_value=10_000_000
            ),
            ColumnSpec(
                "pu_pickup_count_6h", "long", nullable=False, min_value=0, max_value=10_000_000
            ),
            ColumnSpec(
                "pu_pickup_count_24h", "long", nullable=False, min_value=0, max_value=10_000_000
            ),
            ColumnSpec("pu_mean_fare_1h", "double", min_value=-1000, max_value=5000),
            ColumnSpec("pu_mean_fare_6h", "double", min_value=-1000, max_value=5000),
            ColumnSpec("pu_mean_fare_24h", "double", min_value=-1000, max_value=5000),
        ),
    ),
    "silver.driver_trip_history": TableContract(
        table="silver.driver_trip_history",
        columns=(
            ColumnSpec("vendor_id", "long", nullable=False, min_value=1, max_value=6),
            ColumnSpec("pickup_datetime", "timestamp", nullable=False),
            ColumnSpec("event_date", "date", nullable=False),
            ColumnSpec(
                "driver_trip_count_7d", "long", nullable=False, min_value=0, max_value=10_000_000
            ),
            ColumnSpec(
                "driver_trip_count_30d", "long", nullable=False, min_value=0, max_value=10_000_000
            ),
            ColumnSpec("driver_mean_trip_distance_7d", "double", min_value=0, max_value=5000),
            ColumnSpec("driver_mean_trip_distance_30d", "double", min_value=0, max_value=5000),
            ColumnSpec("driver_mean_tip_ratio_7d", "double", min_value=0, max_value=1),
            ColumnSpec("driver_mean_tip_ratio_30d", "double", min_value=0, max_value=1),
        ),
        primary_key=("vendor_id", "event_date"),
    ),
    "silver.trip_features": TableContract(
        table="silver.trip_features",
        columns=(
            ColumnSpec("vendor_id", "long", min_value=1, max_value=6),
            ColumnSpec("pu_location_id", "long", min_value=1, max_value=265),
            ColumnSpec("pickup_datetime", "timestamp", nullable=False),
            ColumnSpec("dropoff_datetime", "timestamp"),
            ColumnSpec("passenger_count", "long", min_value=0, max_value=9),
            ColumnSpec("trip_distance", "double", min_value=0, max_value=5000),
            ColumnSpec("payment_type", "long", min_value=0, max_value=6),
            ColumnSpec("fare_amount", "double", min_value=-1000, max_value=5000),
            ColumnSpec("tip_amount", "double", min_value=-200, max_value=5000),
            ColumnSpec("tip_pct", "double", min_value=0, max_value=1),
            ColumnSpec("event_date", "date", nullable=False),
        ),
    ),
}


def check_contract(df: DataFrame, contract: TableContract) -> list[str]:
    """Return a list of contract violations (empty = contract holds)."""
    violations: list[str] = []
    actual = {field.name: field for field in df.schema.fields}
    for spec in contract.columns:
        field = actual.get(spec.name)
        if field is None:
            violations.append(f"missing column '{spec.name}'")
            continue
        expected = SPARK_TYPES[spec.dtype]
        if not isinstance(field.dataType, type(expected)):
            violations.append(
                f"column '{spec.name}': expected {spec.dtype}, "
                f"got {field.dataType.simpleString()}"
            )
    if violations:
        return violations

    exprs = []
    for spec in contract.columns:
        col = F.col(spec.name)
        if not spec.nullable:
            exprs.append(F.sum(col.isNull().cast("long")).alias(f"_nulls_{spec.name}"))
        if spec.min_value is not None or spec.max_value is not None:
            out = F.lit(False)
            if spec.min_value is not None:
                out = out | (col < F.lit(spec.min_value))
            if spec.max_value is not None:
                out = out | (col > F.lit(spec.max_value))
            exprs.append(F.sum((col.isNotNull() & out).cast("long")).alias(f"_range_{spec.name}"))
    if exprs:
        row = df.agg(*exprs).first().asDict()
        for spec in contract.columns:
            if not spec.nullable and row.get(f"_nulls_{spec.name}"):
                violations.append(
                    f"column '{spec.name}': {row[f'_nulls_{spec.name}']} nulls "
                    "in non-nullable column"
                )
            range_count = row.get(f"_range_{spec.name}")
            if range_count:
                violations.append(
                    f"column '{spec.name}': {range_count} values outside "
                    f"[{spec.min_value}, {spec.max_value}]"
                )
    return violations


def enforce_contract(df: DataFrame, contract: TableContract) -> None:
    """Raise DataQualityError unless ``df`` satisfies ``contract``."""
    violations = check_contract(df, contract)
    if violations:
        raise DataQualityError(f"contract violation for {contract.table}: " + "; ".join(violations))
    log.info("Contract passed for %s", contract.table)
