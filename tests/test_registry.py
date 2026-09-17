"""Tests for features.registry (no live database required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from features import registry


def test_ddl_creates_feature_runs_table() -> None:
    assert "CREATE TABLE IF NOT EXISTS feature_runs" in registry.DDL
    for col in (
        "run_id",
        "feature_group",
        "row_count",
        "input_paths",
        "output_path",
        "spark_conf_json",
        "git_sha",
        "started_at",
        "finished_at",
        "status",
    ):
        assert col in registry.DDL


def test_start_run_inserts_running_row() -> None:
    conn = MagicMock()
    run_id = registry.start_run(
        conn,
        "pickup_zone_demand",
        ["/data/bronze/trips"],
        "/data/silver/pickup_zone_demand",
        {"spark.master": "local[*]"},
        run_key="key-123",
    )
    assert run_id
    cur = conn.cursor.return_value.__enter__.return_value
    sql, params = cur.execute.call_args[0]
    assert "INSERT INTO feature_runs" in sql
    assert params[1] == "pickup_zone_demand"
    assert params[2] is None  # row_count not known yet
    assert json.loads(params[3]) == ["/data/bronze/trips"]
    assert params[4] == "/data/silver/pickup_zone_demand"
    assert json.loads(params[5]) == {"spark.master": "local[*]"}
    assert params[7] is not None  # started_at
    assert params[8] is None  # finished_at
    assert params[9] == "running"
    assert params[10] == "key-123"  # run_key
    conn.commit.assert_called()


def test_is_completed_checks_run_key() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (1,)
    assert registry.is_completed(conn, "key-123") is True
    cur.fetchone.return_value = None
    assert registry.is_completed(conn, "key-123") is False
    assert registry.is_completed(conn, None) is False


def test_supersede_marks_previous_success() -> None:
    conn = MagicMock()
    registry.supersede(conn, "key-123")
    cur = conn.cursor.return_value.__enter__.return_value
    sql, params = cur.execute.call_args[0]
    assert "UPDATE feature_runs" in sql
    assert "superseded" in sql
    assert params[1] == "key-123"


def test_migrations_add_run_key_and_unique_index() -> None:
    assert any("ADD COLUMN IF NOT EXISTS run_key" in stmt for stmt in registry.MIGRATIONS)
    assert any(
        "CREATE UNIQUE INDEX" in stmt and "status = 'success'" in stmt
        for stmt in registry.MIGRATIONS
    )


def test_finish_run_updates_status_and_row_count() -> None:
    conn = MagicMock()
    run_id = registry.start_run(conn, "trip_features", [], "/out", {})
    registry.finish_run(conn, run_id, "success", 42)
    cur = conn.cursor.return_value.__enter__.return_value
    sql, params = cur.execute.call_args[0]
    assert "UPDATE feature_runs" in sql
    assert params[0] == "success"
    assert params[1] == 42
    assert params[2] is not None  # finished_at
    assert params[3] == run_id
