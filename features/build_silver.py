"""Silver feature-layer build.

Computes the feature groups defined in conf/features.yaml from the bronze
taxi table:

* pickup_zone_demand  - per PULocationID trailing pickup counts / mean fare
                        (1h / 6h / 24h, as of each pickup event)
* driver_trip_history - per vendor/day trailing aggregates
                        (7d / 30d: trip count, mean trip distance,
                        mean tip ratio)
* trip_features       - per-trip join of the two groups + tip_pct label

Point-in-time correctness: every aggregate for a row at time T uses only
records with event_ts < T (strict trailing frames via rangeBetween; see
features/point_in_time.py). Output is partitioned Parquet under
silver/<group>/event_date=... and every run is registered in Postgres
(feature_runs).

Run on the docker cluster:
    spark-submit --master spark://spark-master:7077 \
        features/build_silver.py --env docker --profile cluster
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import yaml  # noqa: E402
from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F

from common import atomic, retry  # noqa: E402
from common.config import CONF_DIR, load_config, resolve_path  # noqa: E402
from common.io import read_parquet, table_path  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402
from common.spark import build_spark  # noqa: E402
from features import point_in_time as pit  # noqa: E402
from features import registry  # noqa: E402

log = get_logger(__name__)

EVENT_DATE_COL = "event_date"
TS_UNIX_COL = "event_ts_unix"
LABEL_COL = "tip_pct"

FEATURES_CONFIG = CONF_DIR / "features.yaml"


def load_feature_config() -> dict:
    with FEATURES_CONFIG.open() as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_feature_output(cfg: dict, output: str) -> str:
    """Map a features.yaml output (silver/<group>) onto the env's silver base path."""
    base = str(cfg["paths"]["silver"]).rstrip("/")
    rel = output[len("silver/") :] if output.startswith("silver/") else output
    return f"{base}/{rel}"


def add_tip_pct(df: DataFrame) -> DataFrame:
    """tip_amount / fare_amount clipped to [0, 1]; NULL when fare <= 0."""
    ratio = F.when(F.col("fare_amount") > 0, F.col("tip_amount") / F.col("fare_amount"))
    clipped = F.when(ratio < 0, 0.0).when(ratio > 1, 1.0).otherwise(ratio)
    return df.withColumn(LABEL_COL, clipped)


def windows_from_spec(spec: dict) -> list[tuple[str, int, int]]:
    return [(w["name"], int(w["start_offset"]), int(w["end_offset"])) for w in spec["windows"]]


def aggregations_from_spec(spec: dict) -> list[tuple[str, str, str]]:
    return [(a["prefix"], a["fn"], a["col"]) for a in spec["aggregations"]]


def feature_column_names(spec: dict) -> list[str]:
    return [
        f"{prefix}_{name}"
        for prefix, _, _ in aggregations_from_spec(spec)
        for name, _, _ in windows_from_spec(spec)
    ]


def compute_pickup_zone_demand(bronze: DataFrame, spec: dict) -> DataFrame:
    """Per-PULocationID trailing counts / mean fare, computed as of each pickup event."""
    keys = spec["entity_key"]
    ts_col = spec["event_ts_col"]
    base = bronze.select(*keys, ts_col, EVENT_DATE_COL, "fare_amount")
    for k in keys:
        base = base.filter(F.col(k).isNotNull())
    base = base.filter(F.col(ts_col).isNotNull())
    base = base.withColumn(TS_UNIX_COL, F.unix_timestamp(F.col(ts_col)).cast("long"))
    with_windows = pit.add_trailing_windows(
        base, keys, TS_UNIX_COL, windows_from_spec(spec), aggregations_from_spec(spec)
    )
    return with_windows.select(*keys, ts_col, EVENT_DATE_COL, *feature_column_names(spec))


# Daily partial aggregates used to compute trailing means as sum/count over
# the day-grain window (exact for means, avoids skew from few vendors).
_DAILY_PARTIALS = {
    "trip_distance": ("_dist_cnt", "_dist_sum"),
    "tip_pct": ("_tip_cnt", "_tip_sum"),
}


def compute_driver_trip_history(bronze: DataFrame, spec: dict) -> DataFrame:
    """Per vendor/day trailing aggregates (7d / 30d), day grain.

    Aggregates for day D use only trips with pickup < start of day D
    (frame end offset -86400 on day-grain unix timestamps). The trip-level
    data is first rolled up to (vendor, day) partial sums/counts, then
    trailing windows (rangeBetween) run over ~730 daily rows instead of
    millions of trips -- same math, no vendor skew.
    """
    keys = spec["entity_key"]
    ts_col = spec["event_ts_col"]
    base = bronze.select(
        *keys, ts_col, EVENT_DATE_COL, "trip_distance", "fare_amount", "tip_amount"
    )
    base = add_tip_pct(base)
    for k in keys:
        base = base.filter(F.col(k).isNotNull())
    base = base.filter(F.col(ts_col).isNotNull())

    daily = base.groupBy(*keys, EVENT_DATE_COL).agg(
        F.count(F.lit(1)).alias("_cnt"),
        F.count(F.col("trip_distance")).alias("_dist_cnt"),
        F.sum(F.col("trip_distance")).alias("_dist_sum"),
        F.count(LABEL_COL).alias("_tip_cnt"),
        F.sum(LABEL_COL).alias("_tip_sum"),
    )
    daily = daily.withColumn(TS_UNIX_COL, F.unix_timestamp(F.col(EVENT_DATE_COL)).cast("long"))

    out = daily
    for prefix, fn, col in aggregations_from_spec(spec):
        for name, start_offset, end_offset in windows_from_spec(spec):
            window = pit.trailing_window(keys, TS_UNIX_COL, start_offset, end_offset)
            if fn == "count":
                value = F.coalesce(F.sum("_cnt").over(window), F.lit(0))
            else:
                cnt_col, sum_col = _DAILY_PARTIALS[col]
                value = F.sum(sum_col).over(window) / F.sum(cnt_col).over(window)
            out = out.withColumn(f"{prefix}_{name}", value)

    # The as-of timestamp of a daily feature row is the start of its day, so
    # a trip at any time on day D matches history(D) but never history(D+1).
    day_start = F.to_timestamp(F.col(EVENT_DATE_COL)).alias(ts_col)
    return out.select(*keys, day_start, EVENT_DATE_COL, *feature_column_names(spec))


def compute_trip_features(
    bronze: DataFrame, zone_demand: DataFrame, driver_history: DataFrame, spec: dict
) -> DataFrame:
    """Per-trip spine with the tip_pct label and as-of-joined group features."""
    ts_col = spec["event_ts_col"]
    spine = bronze.select(
        "vendor_id",
        "pu_location_id",
        ts_col,
        "dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "payment_type",
        "fare_amount",
        "tip_amount",
        EVENT_DATE_COL,
    )
    out = add_tip_pct(spine)
    for join_spec in spec["joins"]:
        source = {
            "pickup_zone_demand": zone_demand,
            "driver_trip_history": driver_history,
        }[join_spec["feature_group"]]
        out = pit.as_of_join(
            out,
            source,
            ts_col=ts_col,
            keys=join_spec["keys"],
            tolerance=int(join_spec["tolerance_seconds"]),
            prefix=join_spec["prefix"],
        )
    return out


def group_run_key(feature_group: str, input_paths: list[str], git_sha: str) -> str:
    """Deterministic identity of a logical run: group + code version + inputs."""
    digest = hashlib.sha256()
    digest.update(feature_group.encode())
    digest.update((git_sha or "uncommitted").encode())
    for p in sorted(input_paths):
        digest.update(p.encode())
    return digest.hexdigest()[:32]


def build_and_register(
    spark,
    conn,
    name: str,
    df: DataFrame,
    input_paths: list[str],
    output: str,
    run_key: str,
    force: bool = False,
) -> int | None:
    """Count, write via staging + atomic promote, and record the run.

    A completed ``run_key`` makes this a no-op unless ``force`` is set.
    Writes never leave a half-written partition visible: data lands in a
    staging dir and is promoted partition-by-partition (see common/atomic).
    """
    output_path = resolve_path(output)
    spark_conf = dict(spark.sparkContext.getConf().getAll())
    if conn and run_key and not force and registry.is_completed(conn, run_key):
        log.info(
            "Feature group '%s': completed run exists (run_key=%s) -- no-op; "
            "pass --force to rebuild",
            name,
            run_key,
        )
        return None
    if conn and run_key and force:
        registry.supersede(conn, run_key)

    run_id = None
    if conn:
        run_id = retry.retry(
            lambda: registry.start_run(conn, name, input_paths, output_path, spark_conf, run_key)
        )
    try:
        df.cache()
        n = retry.retry(lambda: df.count())
        retry.retry(
            lambda: atomic.atomic_write_parquet(
                spark,
                df,
                output,
                partition_by=[EVENT_DATE_COL],
                run_id=run_id or uuid.uuid4().hex,
            )
        )
        if conn:
            retry.retry(lambda: registry.finish_run(conn, run_id, "success", n))
        log.info("Feature group '%s': %d rows -> %s", name, n, output_path)
        return n
    except Exception:
        if conn:
            try:
                retry.retry(lambda: registry.finish_run(conn, run_id, "failed", None))
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not record failed run %s: %s", run_id, exc)
        raise
    finally:
        df.unpersist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default=None, help="Environment config (conf/<env>.yaml)")
    parser.add_argument("--profile", default=None, help="Spark profile (local|cluster|k8s)")
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated subset of feature groups to build (default: all)",
    )
    parser.add_argument(
        "--no-registry", action="store_true", help="Skip Postgres feature_runs registration"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when a completed run exists for the same run_key",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    cfg = load_config(args.env)
    feat_cfg = load_feature_config()
    groups: dict = feat_cfg["feature_groups"]
    if args.groups:
        wanted = set(args.groups.split(","))
        missing = wanted - set(groups)
        if missing:
            raise SystemExit(f"Unknown feature groups: {sorted(missing)}")
    else:
        wanted = set(groups)

    spark = build_spark("silver-features", env=args.env, profile=args.profile)

    conn = None
    if not args.no_registry:
        conn = retry.retry(lambda: registry.connect(env=args.env))
        registry.ensure_table(conn)
        log.info("Connected to Postgres feature_runs registry")

    bronze_path = resolve_path(table_path(cfg["paths"]["bronze"], "trips"))
    log.info("Reading bronze trips from %s", bronze_path)
    bronze = read_parquet(spark, bronze_path)

    git_sha = registry.current_git_sha() or "uncommitted"
    counts: dict[str, object] = {}

    demand_spec = groups["pickup_zone_demand"]
    demand_output = resolve_feature_output(cfg, demand_spec["output"])
    demand_path = resolve_path(demand_output)
    demand_key = group_run_key("pickup_zone_demand", [bronze_path], git_sha)
    demand = None
    if "pickup_zone_demand" in wanted:
        if conn and not args.force and registry.is_completed(conn, demand_key):
            log.info("pickup_zone_demand: skipping (completed run, run_key=%s)", demand_key)
            counts["pickup_zone_demand"] = "skipped (no-op)"
        else:
            demand = compute_pickup_zone_demand(bronze, demand_spec)
            counts["pickup_zone_demand"] = build_and_register(
                spark,
                conn,
                "pickup_zone_demand",
                demand,
                [bronze_path],
                demand_output,
                demand_key,
                args.force,
            )

    driver_spec = groups["driver_trip_history"]
    driver_output = resolve_feature_output(cfg, driver_spec["output"])
    driver_path = resolve_path(driver_output)
    driver_key = group_run_key("driver_trip_history", [bronze_path], git_sha)
    driver = None
    if "driver_trip_history" in wanted:
        if conn and not args.force and registry.is_completed(conn, driver_key):
            log.info("driver_trip_history: skipping (completed run, run_key=%s)", driver_key)
            counts["driver_trip_history"] = "skipped (no-op)"
        else:
            driver = compute_driver_trip_history(bronze, driver_spec)
            counts["driver_trip_history"] = build_and_register(
                spark,
                conn,
                "driver_trip_history",
                driver,
                [bronze_path],
                driver_output,
                driver_key,
                args.force,
            )

    if "trip_features" in wanted:
        trip_spec = groups["trip_features"]
        trip_output = resolve_feature_output(cfg, trip_spec["output"])
        trip_key = group_run_key("trip_features", [bronze_path, demand_path, driver_path], git_sha)
        if conn and not args.force and registry.is_completed(conn, trip_key):
            log.info("trip_features: skipping (completed run, run_key=%s)", trip_key)
            counts["trip_features"] = "skipped (no-op)"
        else:
            demand_df = demand if demand is not None else read_parquet(spark, demand_path)
            driver_df = driver if driver is not None else read_parquet(spark, driver_path)
            trips = compute_trip_features(bronze, demand_df, driver_df, trip_spec)
            trips.printSchema()
            counts["trip_features"] = build_and_register(
                spark,
                conn,
                "trip_features",
                trips,
                [bronze_path, demand_path, driver_path],
                trip_output,
                trip_key,
                args.force,
            )

    log.info("=" * 64)
    log.info("Silver build complete. Row counts per feature group:")
    for name, n in counts.items():
        log.info("  %-20s %s", name, n)

    if conn:
        conn.close()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
