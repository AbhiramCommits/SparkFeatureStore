"""One configuration point of the shuffle-partitions x AQE sweep.

Runs the real pickup_zone_demand computation (1h/6h/24h trailing windows
over the bronze trips table) with a fresh Spark session for each
(partitions, aqe) combination. Records wall time, max task duration and
shuffle read bytes from the Spark REST API and appends one row to
bench/results/partition_sweep.csv.

Sweep all 8 combinations via: make bench-partitions
Single run:
    spark-submit --master spark://spark-master:7077 \
        bench/partition_sweep.py --partitions 200 --aqe true
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
from features.build_silver import compute_pickup_zone_demand, load_feature_config  # noqa: E402

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "bench" / "results"
CSV_COLUMNS = [
    "ts",
    "partitions",
    "aqe",
    "wall_s",
    "max_task_ms",
    "shuffle_read_bytes",
    "stages",
    "rows",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--partitions", type=int, required=True, help="spark.sql.shuffle.partitions"
    )
    parser.add_argument(
        "--aqe", choices=["true", "false"], required=True, help="Adaptive query execution on/off"
    )
    parser.add_argument("--env", default=None, help="Environment config (conf/<env>.yaml)")
    parser.add_argument("--profile", default=None, help="Spark profile")
    parser.add_argument("--rest-base", default="http://localhost:4040", help="Spark REST API base")
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "partition_sweep.csv"), help="Results CSV"
    )
    return parser


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    aqe_enabled = args.aqe == "true"
    spark = build_spark(
        "partition-sweep",
        env=args.env,
        profile=args.profile,
        conf_overrides={
            "spark.sql.shuffle.partitions": str(args.partitions),
            "spark.sql.adaptive.enabled": str(aqe_enabled).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(aqe_enabled).lower(),
        },
    )

    cfg = load_config(args.env)
    spec = load_feature_config()["feature_groups"]["pickup_zone_demand"]
    bronze = read_parquet(spark, table_path(cfg["paths"]["bronze"], "trips"))
    df = compute_pickup_zone_demand(bronze, spec)

    watermark = max_completed_stage_id(spark, args.rest_base)
    t0 = time.perf_counter()
    # Aggregate over the window feature columns so the optimizer cannot
    # prune the trailing-window computation out of the plan.
    row = df.agg(
        F.count(F.col("pu_pickup_count_1h")),
        F.sum(F.col("pu_pickup_count_24h")),
    ).collect()[0]
    wall = time.perf_counter() - t0
    metrics = collect_stage_metrics(spark, args.rest_base, watermark)

    result = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "partitions": args.partitions,
        "aqe": args.aqe,
        "wall_s": round(wall, 2),
        "max_task_ms": metrics["max_task_ms"],
        "shuffle_read_bytes": metrics["shuffle_read_bytes"],
        "stages": metrics["stages"],
        "rows": int(row[0]),
    }
    append_result(Path(args.out), result)
    log.info(
        "partitions=%s aqe=%s: wall=%.2fs max_task=%sms shuffle_read=%d bytes stages=%d rows=%d",
        args.partitions,
        args.aqe,
        wall,
        metrics["max_task_ms"],
        metrics["shuffle_read_bytes"],
        metrics["stages"],
        int(row[0]),
    )
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
