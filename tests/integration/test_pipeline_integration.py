"""End-to-end integration test: the docker-compose Spark cluster runs
bronze -> silver, then sklearn trains on the silver tables.

Preconditions (see Makefile): `make up` (stack running) and `make fetch`
(one month of raw data, default Jan 2023). Run explicitly with
`make integration` -- too slow for the default unit suite.

Expected row counts are for the default 1-month slice
(yellow_tripdata_2023-01.parquet).
"""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.integration

EXPECTED_ROWS = {
    "bronze/trips": 3_066_766,
    "silver/pickup_zone_demand": 3_066_766,
    "silver/driver_trip_history": 67,
    "silver/trip_features": 3_066_766,
}
REPO = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def _spark_submit(script: str, *args: str) -> subprocess.CompletedProcess:
    return _run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "spark-master",
            "/opt/spark/bin/spark-submit",
            "--master",
            "spark://spark-master:7077",
            "--deploy-mode",
            "client",
            f"/opt/sparkfeaturestore/{script}",
            "--env",
            "docker",
            "--profile",
            "cluster",
            *args,
        ],
        timeout=3600,
    )


def _rows_in(table_rel: str) -> int:
    pattern = str(REPO / "data" / table_rel / "event_date=*" / "part-*.parquet")
    files = glob.glob(pattern)
    assert files, f"no partitions found for {table_rel}"
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


@pytest.fixture(scope="module")
def stack():
    """Fail early with a clear message when the preconditions are missing."""
    check = subprocess.run(
        ["docker", "compose", "ps", "--status", "running", "spark-master"],
        capture_output=True,
        text=True,
    )
    if "spark-master" not in check.stdout:
        pytest.skip("docker-compose stack is not running -- run 'make up' first")
    if not list((REPO / "data" / "raw").glob("yellow_tripdata_*.parquet")):
        pytest.skip("no raw parquet in data/raw -- run 'make fetch' first")
    return True


def test_bronze_row_counts(stack) -> None:
    _spark_submit("ingest/bronze.py")
    assert _rows_in("bronze/trips") == EXPECTED_ROWS["bronze/trips"]


def test_silver_row_counts(stack) -> None:
    # --force: the integration test rebuilds regardless of prior run_keys
    _spark_submit("features/build_silver.py", "--force")
    for table, expected in EXPECTED_ROWS.items():
        if table.startswith("silver/"):
            assert _rows_in(table) == expected, table


def test_feature_runs_registered_in_postgres(stack) -> None:
    from training import artifacts

    conn = artifacts.connect(env="local")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT feature_group, status, row_count FROM feature_runs "
                "WHERE status = 'success' ORDER BY finished_at DESC LIMIT 3"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    by_group = {group: (status, count) for group, status, count in rows}
    assert set(by_group) == {
        "pickup_zone_demand",
        "driver_trip_history",
        "trip_features",
    }
    for status, count in by_group.values():
        assert status == "success"
        assert count > 0


def test_sklearn_train_beats_baseline_and_registers(stack) -> None:
    _run(
        ["uv", "run", "python", "training/sklearn_job.py", "--env", "local"],
        cwd=REPO,
        timeout=1800,
    )
    metric_files = sorted((REPO / "artifacts" / "sklearn").glob("*/metrics.json"))
    assert metric_files, "no sklearn artifacts found"
    metrics = json.loads(metric_files[-1].read_text())
    assert metrics["row_counts"] == {"train": 1_826_432, "val": 500_407, "test": 713_721}
    test, baseline = metrics["test"], metrics["baseline"]
    assert test["rmse"] < baseline["rmse"], "model must beat the mean baseline"
    assert test["mae"] < baseline["mae"]
    assert test["r2"] > 0

    from training import artifacts

    conn = artifacts.connect(env="local")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM model_runs WHERE framework = 'sklearn'")
            assert cur.fetchone()[0] >= 1
    finally:
        conn.close()
