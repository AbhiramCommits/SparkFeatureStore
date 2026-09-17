"""Benchmark: skewed groupBy aggregation with vs without hot-key salting.

Runs the same aggregation twice on the bronze trips table -- plain, then
salted two-phase (features/skew.py) -- and records wall time, max task
duration and shuffle read bytes from the Spark REST API. Results are
appended to bench/results/skew_bench.csv.

Run inside the docker cluster:
    spark-submit --master spark://spark-master:7077 \
        bench/skew_bench.py --env docker
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402

from bench.metrics import collect_stage_metrics, max_completed_stage_id  # noqa: E402
from common.config import REPO_ROOT, load_config  # noqa: E402
from common.io import read_parquet, table_path  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402
from common.spark import build_spark  # noqa: E402
from features.skew import add_salt, detect_hot_keys, salted_aggregate  # noqa: E402

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "bench" / "results"
CSV_COLUMNS = [
    "ts",
    "mode",
    "rows",
    "hot_keys",
    "salt_buckets",
    "wall_s",
    "max_task_ms",
    "shuffle_read_bytes",
    "stages",
]

AGGREGATES = [("fare_amount", "avg"), ("fare_amount", "sum"), ("fare_amount", "count")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default=None, help="Environment config (conf/<env>.yaml)")
    parser.add_argument("--profile", default=None, help="Spark profile")
    parser.add_argument("--threshold", type=int, default=50000, help="Hot-key row-count threshold")
    parser.add_argument("--salt-buckets", type=int, default=16, help="Salt shards per hot key")
    parser.add_argument("--rest-base", default="http://localhost:4040", help="Spark REST API base")
    parser.add_argument("--out", default=str(RESULTS_DIR / "skew_bench.csv"), help="Results CSV")
    return parser


def append_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    spark = build_spark("skew-bench", env=args.env, profile=args.profile)
    cfg = load_config(args.env)
    bronze = read_parquet(spark, table_path(cfg["paths"]["bronze"], "trips")).filter(
        F.col("pu_location_id").isNotNull()
    )
    rows = bronze.count()
    hot = detect_hot_keys(bronze, "pu_location_id", threshold=args.threshold)
    log.info(
        "Input: %d rows; %d hot PULocationIDs above threshold %d", rows, len(hot), args.threshold
    )

    results: list[dict] = []

    def _run(mode: str, df) -> None:
        watermark = max_completed_stage_id(spark, args.rest_base)
        t0 = time.perf_counter()
        n = df.count()
        wall = time.perf_counter() - t0
        metrics = collect_stage_metrics(spark, args.rest_base, watermark)
        results.append(
            {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "mode": mode,
                "rows": n,
                "hot_keys": len(hot),
                "salt_buckets": args.salt_buckets,
                "wall_s": round(wall, 2),
                "max_task_ms": metrics["max_task_ms"],
                "shuffle_read_bytes": metrics["shuffle_read_bytes"],
                "stages": metrics["stages"],
            }
        )

    # --- 1. groupBy aggregation: unsalted vs salted ---
    plain = bronze.groupBy("pu_location_id").agg(
        F.avg("fare_amount").alias("fare_amount_avg"),
        F.sum("fare_amount").alias("fare_amount_sum"),
        F.count(F.lit(1)).alias("fare_amount_count"),
    )
    salted = salted_aggregate(
        bronze,
        group_cols=["pu_location_id"],
        hot_col="pu_location_id",
        hot_keys=hot,
        salt_buckets=args.salt_buckets,
        aggregates=AGGREGATES,
    )
    _run("unsalted_groupby", plain)
    _run("salted_groupby", salted)

    # --- 2. full-row keyed shuffle (window/join pattern): unsalted vs salted ---
    # Salting spreads hot keys over salt buckets in the first shuffle, then
    # re-keys in a second shuffle: two balanced shuffles instead of one
    # skewed shuffle.
    unsalted_shuffle = bronze.repartition(16, "pu_location_id")
    salted_shuffle = (
        add_salt(bronze, "pu_location_id", hot, args.salt_buckets)
        .repartition(16, "pu_location_id", "_salt")
        .drop("_salt")
        .repartition(16, "pu_location_id")
    )
    _run("unsalted_shuffle", unsalted_shuffle)
    _run("salted_shuffle", salted_shuffle)

    # --- correctness: salting must not change the groupBy result ---
    plain_map = {
        r.pu_location_id: (r.fare_amount_count, r.fare_amount_sum, r.fare_amount_avg)
        for r in plain.collect()
    }
    salted_map = {
        r.pu_location_id: (r.fare_amount_count, r.fare_amount_sum, r.fare_amount_avg)
        for r in salted.collect()
    }
    assert set(plain_map) == set(salted_map), "key sets differ"
    for key, (pc, ps, pa) in plain_map.items():
        sc, ss, sa = salted_map[key]
        assert pc == sc, f"count mismatch for {key}"
        # sums may differ in the last ULP due to summation order across
        # salted partial aggregates; counts must be exact.
        assert abs((ps or 0.0) - (ss or 0.0)) <= 1e-6 * max(
            abs(ps or 0.0), 1.0
        ), f"sum mismatch for {key}"
        assert abs((pa or 0.0) - (sa or 0.0)) < 1e-6, f"avg mismatch for {key}"
    log.info("Equivalence check passed: salted == unsalted across %d keys", len(plain_map))

    append_results(Path(args.out), results)
    log.info("=" * 64)
    log.info("Skew benchmark results (appended to %s):", args.out)
    for row in results:
        log.info(
            "  %-9s wall=%ss max_task=%sms shuffle_read=%d bytes stages=%d",
            row["mode"],
            row["wall_s"],
            row["max_task_ms"],
            row["shuffle_read_bytes"],
            row["stages"],
        )
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
