"""Per-run versioned training artifacts + the Postgres model_runs registry.

Every training run writes to ``artifacts/<framework>/<run_id>/``:

* model file (joblib for sklearn, state_dict for torch)
* metrics.json, features.json, hyperparams.json, metadata.json (git sha,
  feature_run_ids the model was trained on, row counts)

and registers a row in ``model_runs`` with a foreign key to the
``feature_runs`` rows that produced the input features.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib

from common.config import REPO_ROOT, load_config
from common.logging import get_logger

log = get_logger(__name__)

ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", REPO_ROOT / "artifacts"))

TABLE_NAME = "model_runs"
JUNCTION_TABLE = "model_run_features"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    run_id UUID PRIMARY KEY,
    framework TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics_json JSONB NOT NULL DEFAULT '{{}}',
    features JSONB NOT NULL DEFAULT '[]',
    hyperparams_json JSONB NOT NULL DEFAULT '{{}}',
    git_sha TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

# Real FK to feature_runs via a junction table (Postgres does not support
# array -> scalar foreign keys).
JUNCTION_DDL = f"""
CREATE TABLE IF NOT EXISTS {JUNCTION_TABLE} (
    model_run_id UUID NOT NULL REFERENCES {TABLE_NAME}(run_id) ON DELETE CASCADE,
    feature_run_id UUID NOT NULL REFERENCES feature_runs(run_id),
    PRIMARY KEY (model_run_id, feature_run_id)
)
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (run_id, framework, artifact_path, metrics_json, features, hyperparams_json,
     git_sha, created_at)
VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
"""

INSERT_JUNCTION_SQL = f"""
INSERT INTO {JUNCTION_TABLE} (model_run_id, feature_run_id)
VALUES (%s, %s)
"""

LATEST_FEATURE_RUNS_SQL = """
SELECT DISTINCT ON (feature_group) run_id
FROM feature_runs
WHERE feature_group = ANY(%s) AND status = 'success'
ORDER BY feature_group, finished_at DESC
"""


class ArtifactRun:
    def __init__(self, framework: str, run_id: str | None = None):
        self.framework = framework
        self.run_id = run_id or uuid.uuid4().hex
        self.path = ARTIFACTS_ROOT / framework / self.run_id

    def ensure(self) -> "ArtifactRun":
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def write_json(self, name: str, payload: dict | list) -> Path:
        target = self.path / f"{name}.json"
        target.write_text(json.dumps(payload, indent=2, default=str))
        return target


def save_model(run: ArtifactRun, model, framework: str) -> Path:
    if framework == "sklearn":
        target = run.path / "model.joblib"
        joblib.dump(model, target)
    elif framework == "torch":
        target = run.path / "model_state.pt"
        import torch

        torch.save(model.state_dict(), target)
    else:
        raise ValueError(f"unknown framework: {framework}")
    return target


def write_metadata(
    run: ArtifactRun,
    *,
    features: list[str],
    hyperparams: dict,
    git_sha: str,
    feature_run_ids: list[str],
    row_counts: dict[str, int],
) -> Path:
    return run.write_json(
        "metadata",
        {
            "framework": run.framework,
            "run_id": run.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "feature_run_ids": feature_run_ids,
            "features": features,
            "hyperparams": hyperparams,
            "row_counts": row_counts,
        },
    )


def connect(env: str | None = None):
    """Open a Postgres connection using conf/<env>.yaml (postgres section)."""
    import psycopg

    pg = load_config(env).get("postgres", {})
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
        cur.execute(JUNCTION_DDL)
    conn.commit()


def fetch_latest_feature_runs(conn, feature_groups: list[str]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(LATEST_FEATURE_RUNS_SQL, (feature_groups,))
        return [str(row[0]) for row in cur.fetchall()]


def register_model_run(
    conn,
    *,
    run: ArtifactRun,
    framework: str,
    metrics: dict,
    features: list[str],
    hyperparams: dict,
    git_sha: str,
    feature_run_ids: list[str],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            (
                run.run_id,
                framework,
                str(run.path),
                json.dumps(metrics),
                json.dumps(features),
                json.dumps(hyperparams),
                git_sha,
                datetime.now(timezone.utc),
            ),
        )
        for feature_run_id in feature_run_ids:
            cur.execute(INSERT_JUNCTION_SQL, (run.run_id, feature_run_id))
    conn.commit()
    log.info("Registered model run %s (%s) in %s", run.run_id, framework, TABLE_NAME)
