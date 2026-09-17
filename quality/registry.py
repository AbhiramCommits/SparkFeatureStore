"""Postgres registry for data-quality results.

One row per gate execution with status, measured value and threshold --
so every failed gate that blocked a write is auditable.
"""

from __future__ import annotations

from datetime import datetime

from common.logging import get_logger

log = get_logger(__name__)

TABLE_NAME = "quality_results"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES feature_runs(run_id),
    feature_group TEXT NOT NULL,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    measured_value TEXT,
    threshold TEXT,
    message TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (run_id, feature_group, check_name, status, measured_value, threshold, message, checked_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def record_result(
    conn,
    *,
    run_id: str | None,
    feature_group: str,
    check_name: str,
    status: str,
    measured_value: str | None = None,
    threshold: str | None = None,
    message: str | None = None,
    checked_at: datetime | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            (
                run_id,
                feature_group,
                check_name,
                status,
                measured_value,
                threshold,
                message,
                checked_at or datetime.now(),
            ),
        )
    conn.commit()
    log.debug("Recorded quality result %s/%s = %s", feature_group, check_name, status)
