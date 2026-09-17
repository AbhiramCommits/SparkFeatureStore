"""Spark REST API helpers for benchmarks.

Pulls stage metrics from the Spark REST API (driver UI or master proxy) at
``/api/v1/applications/<app-id>/stages``: max task duration (from the
``executorRunTime`` task-summary quantile 1.0) and shuffle read bytes.
"""

from __future__ import annotations

import json
import urllib.request

from pyspark.sql import SparkSession

from common.logging import get_logger

log = get_logger(__name__)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def _stages_url(spark: SparkSession, rest_base: str) -> str:
    app_id = spark.sparkContext.applicationId
    return f"{rest_base}/api/v1/applications/{app_id}/stages"


def max_completed_stage_id(spark: SparkSession, rest_base: str) -> int:
    """Highest stage id seen so far (used as a watermark between runs)."""
    try:
        stages = fetch_json(_stages_url(spark, rest_base))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not query stage list (watermark=0): %s", exc)
        return -1
    return max((st.get("stageId", -1) for st in stages), default=-1)


def collect_stage_metrics(spark: SparkSession, rest_base: str, after_stage_id: int) -> dict:
    """Aggregate metrics for completed stages with id > ``after_stage_id``.

    Shuffle read bytes come from the stage list (stage-level totals);
    max task duration comes from the per-stage taskSummary endpoint
    (``executorRunTime`` quantiles, last element = max).
    Returns {"stages": n, "max_task_ms": ms, "shuffle_read_bytes": bytes}.
    """
    stages = fetch_json(_stages_url(spark, rest_base))
    total_shuffle_read = 0
    max_task_ms = 0
    n_stages = 0
    for st in stages:
        if st.get("stageId", -1) <= after_stage_id:
            continue
        if st.get("status") != "COMPLETE":
            continue
        n_stages += 1
        total_shuffle_read += int(st.get("shuffleReadBytes", 0))
        url = (
            f"{_stages_url(spark, rest_base)}/{st['stageId']}/"
            f"{st.get('attemptId', 0)}/taskSummary"
        )
        try:
            summary = fetch_json(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch task summary for stage %s: %s", st["stageId"], exc)
            continue
        run_times = summary.get("executorRunTime") or []
        if run_times:
            max_task_ms = max(max_task_ms, int(run_times[-1]))
    return {
        "stages": n_stages,
        "max_task_ms": max_task_ms,
        "shuffle_read_bytes": total_shuffle_read,
    }
