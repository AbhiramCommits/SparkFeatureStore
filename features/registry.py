"""Feature-run registry: records silver feature-group builds in Postgres.

One row in ``feature_runs`` per materialized feature-group run with
provenance: inputs, output path, Spark configuration, git SHA, timestamps
and final status.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from common.config import REPO_ROOT, load_config
from common.logging import get_logger

log = get_logger(__name__)

TABLE_NAME = "feature_runs"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    run_id UUID PRIMARY KEY,
    feature_group TEXT NOT NULL,
    row_count BIGINT,
    input_paths JSONB NOT NULL DEFAULT '[]',
    output_path TEXT NOT NULL,
    spark_conf_json JSONB,
    git_sha TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
)
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (run_id, feature_group, row_count, input_paths, output_path, spark_conf_json,
     git_sha, started_at, finished_at, status)
VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s)
"""

FINISH_SQL = f"""
UPDATE {TABLE_NAME} SET status = %s, row_count = %s, finished_at = %s WHERE run_id = %s
"""


def get_pg_config(env: str | None = None) -> dict[str, Any]:
    return load_config(env).get("postgres", {})


def connect(pg: dict[str, Any] | None = None, env: str | None = None):
    """Open a connection using the ``postgres`` section of conf/<env>.yaml."""
    import psycopg

    pg = pg or get_pg_config(env)
    return psycopg.connect(
        host=pg["host"],
        port=int(pg["port"]),
        dbname=pg["database"],
        user=pg["user"],
        password=pg["password"],
        connect_timeout=10,
    )


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def current_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def start_run(
    conn,
    feature_group: str,
    input_paths: list[str],
    output_path: str,
    spark_conf: dict[str, str],
) -> str:
    """Insert a 'running' row and return its run_id."""
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            (
                run_id,
                feature_group,
                None,
                json.dumps(input_paths or []),
                output_path,
                json.dumps(spark_conf or {}),
                current_git_sha(),
                started_at,
                None,
                "running",
            ),
        )
    conn.commit()
    log.info("Registered feature run %s for %s", run_id, feature_group)
    return run_id


def finish_run(conn, run_id: str, status: str, row_count: int | None) -> None:
    """Mark a run as 'success' or 'failed' with its final row count."""
    finished_at = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(FINISH_SQL, (status, row_count, finished_at, run_id))
    conn.commit()
    log.info("Feature run %s finished: %s (rows=%s)", run_id, status, row_count)
