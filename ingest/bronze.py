"""Bronze ingestion job: raw NYC TLC Yellow Taxi Parquet -> bronze layer.

Reads raw trip data, normalizes column names (snake_case) and dtypes, adds
``ingest_ts`` (load timestamp) and ``event_date`` (derived from pickup time),
then writes a Hive-style partitioned Parquet table:

    bronze/trips/event_date=2023-01-01/part-*.parquet

Run on the docker cluster:
    spark-submit --master spark://spark-master:7077 \
        ingest/bronze.py --env docker --profile cluster

Run locally:
    python ingest/bronze.py --env local --profile local
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F

from common import atomic, retry  # noqa: E402
from common.config import load_config, resolve_path  # noqa: E402
from common.io import read_parquet, table_path  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402
from common.spark import build_spark  # noqa: E402

log = get_logger(__name__)

INGEST_TS_COL = "ingest_ts"
EVENT_DATE_COL = "event_date"

# TLC columns that need explicit renaming; everything else falls back to
# snake_case normalization.
RENAME_MAP = {
    "VendorID": "vendor_id",
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "RatecodeID": "rate_code_id",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
}

# Canonical bronze dtypes for the 2023 Yellow Taxi schema.
CAST_MAP = {
    "vendor_id": "long",
    "pickup_datetime": "timestamp",
    "dropoff_datetime": "timestamp",
    "passenger_count": "long",
    "trip_distance": "double",
    "rate_code_id": "long",
    "pu_location_id": "long",
    "do_location_id": "long",
    "payment_type": "long",
    "fare_amount": "double",
    "extra": "double",
    "mta_tax": "double",
    "tip_amount": "double",
    "tolls_amount": "double",
    "improvement_surcharge": "double",
    "total_amount": "double",
    "congestion_surcharge": "double",
    "airport_fee": "double",
    "store_and_fwd_flag": "string",
}

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def snake_case(name: str) -> str:
    """Convert ``CamelCase`` / ``camelCase`` column names to snake_case."""
    return re.sub(r"_+", "_", _CAMEL_RE.sub("_", name)).strip("_").lower()


def normalize_columns(df: DataFrame) -> DataFrame:
    for old, new in RENAME_MAP.items():
        if old in df.columns and new not in df.columns:
            df = df.withColumnRenamed(old, new)
    for name in df.columns:
        normalized = snake_case(name)
        if normalized != name and normalized not in df.columns:
            df = df.withColumnRenamed(name, normalized)
    return df


def cast_columns(df: DataFrame) -> DataFrame:
    for name, dtype in CAST_MAP.items():
        if name in df.columns:
            df = df.withColumn(name, F.col(name).cast(dtype))
    return df


def add_derived_columns(df: DataFrame, event_ts_col: str = "pickup_datetime") -> DataFrame:
    return df.withColumn(INGEST_TS_COL, F.current_timestamp()).withColumn(
        EVENT_DATE_COL, F.to_date(F.col(event_ts_col))
    )


def run(
    env: str | None = None,
    profile: str | None = None,
    raw_path: str | None = None,
    bronze_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> int:
    cfg = load_config(env)
    raw_path = raw_path or cfg["paths"]["raw"]
    bronze_path = bronze_path or table_path(cfg["paths"]["bronze"], "trips")

    spark = build_spark("bronze-trips", env=env, profile=profile)

    df = read_parquet(spark, raw_path)
    raw_rows = df.count()
    log.info("Raw rows read: %d", raw_rows)

    df = add_derived_columns(cast_columns(normalize_columns(df)))
    if start:
        df = df.filter(F.col(EVENT_DATE_COL) >= F.lit(start))
    if end:
        df = df.filter(F.col(EVENT_DATE_COL) <= F.lit(end))

    df.cache()
    bronze_rows = df.count()
    log.info("Bronze rows after transform: %d", bronze_rows)

    # Staging + atomic promotion: downstream readers never see a
    # half-written partition (see common/atomic.py).
    retry.retry(
        lambda: atomic.atomic_write_parquet(
            spark,
            df,
            bronze_path,
            partition_by=[EVENT_DATE_COL],
            run_id=f"bronze-{uuid.uuid4().hex[:12]}",
        )
    )

    written = spark.read.parquet(resolve_path(bronze_path))
    written_rows = written.count()
    partitions = written.select(EVENT_DATE_COL).distinct().orderBy(EVENT_DATE_COL).count()
    log.info(
        "Bronze table at %s: %d rows across %d event_date partitions",
        resolve_path(bronze_path),
        written_rows,
        partitions,
    )
    written.printSchema()
    df.unpersist()
    spark.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env", default=None, help="Environment config (conf/<env>.yaml), default: local"
    )
    parser.add_argument("--profile", default=None, help="Spark profile (local|cluster|k8s)")
    parser.add_argument("--raw-path", default=None, help="Override raw input path")
    parser.add_argument("--bronze-path", default=None, help="Override bronze output path")
    parser.add_argument("--start", default=None, help="Keep rows with event_date >= YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Keep rows with event_date <= YYYY-MM-DD")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return run(
        env=args.env,
        profile=args.profile,
        raw_path=args.raw_path,
        bronze_path=args.bronze_path,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    raise SystemExit(main())
