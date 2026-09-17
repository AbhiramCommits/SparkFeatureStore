"""Tests for training/artifacts.py: versioned artifact directories + the
model_runs registry (no live database required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from training import artifacts


def test_artifact_run_is_versioned_per_framework(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path)
    run = artifacts.ArtifactRun("sklearn").ensure()
    assert run.path == tmp_path / "sklearn" / run.run_id
    assert run.path.is_dir()

    run2 = artifacts.ArtifactRun("sklearn")
    assert run2.run_id != run.run_id  # every run gets a fresh version
    assert run2.path.parent == tmp_path / "sklearn"

    # explicit run_id is honored (reproducible artifact paths)
    run3 = artifacts.ArtifactRun("torch", run_id="fixed-id").ensure()
    assert run3.path == tmp_path / "torch" / "fixed-id"


def test_write_json_roundtrip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path)
    run = artifacts.ArtifactRun("sklearn", run_id="r1").ensure()
    target = run.write_json("metrics", {"test": {"rmse": 0.13}})
    assert target.name == "metrics.json"
    assert json.loads(target.read_text()) == {"test": {"rmse": 0.13}}


def test_write_metadata_contains_provenance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path)
    run = artifacts.ArtifactRun("sklearn", run_id="r1").ensure()
    artifacts.write_metadata(
        run,
        features=["a", "b"],
        hyperparams={"lr": 1e-3},
        git_sha="abc123",
        feature_run_ids=["f1", "f2"],
        row_counts={"train": 100, "val": 20, "test": 30},
    )
    meta = json.loads((run.path / "metadata.json").read_text())
    assert meta["run_id"] == "r1"
    assert meta["git_sha"] == "abc123"
    assert meta["feature_run_ids"] == ["f1", "f2"]
    assert meta["features"] == ["a", "b"]
    assert meta["hyperparams"] == {"lr": 1e-3}
    assert meta["row_counts"] == {"train": 100, "val": 20, "test": 30}


def test_save_model_sklearn_joblib(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path)
    run = artifacts.ArtifactRun("sklearn", run_id="r1").ensure()
    model_path = artifacts.save_model(run, {"weights": [1, 2]}, "sklearn")
    assert model_path.name == "model.joblib"
    import joblib

    assert joblib.load(model_path) == {"weights": [1, 2]}


def test_save_model_rejects_unknown_framework(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts, "ARTIFACTS_ROOT", tmp_path)
    run = artifacts.ArtifactRun("sklearn", run_id="r1").ensure()
    try:
        artifacts.save_model(run, object(), "tensorflow")
    except ValueError as exc:
        assert "tensorflow" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_register_model_run_inserts_and_links_features() -> None:
    conn = MagicMock()
    run = artifacts.ArtifactRun("sklearn", run_id="model-1")
    artifacts.register_model_run(
        conn,
        run=run,
        framework="sklearn",
        metrics={"test": {"rmse": 0.13}},
        features=["a"],
        hyperparams={"lr": 1e-3},
        git_sha="abc",
        feature_run_ids=["f1", "f2"],
    )
    cur = conn.cursor.return_value.__enter__.return_value
    calls = [c.args for c in cur.execute.call_args_list]
    insert = [
        args
        for args in calls
        if "model_runs" in str(args[0]) and "model_run_features" not in str(args[0])
    ]
    junction = [args for args in calls if "model_run_features" in str(args[0])]
    assert len(insert) == 1
    sql, params = insert[0]
    assert params[0] == "model-1"
    assert params[1] == "sklearn"
    assert json.loads(params[3]) == {"test": {"rmse": 0.13}}
    # one junction row per feature_run_id
    assert sorted(args[1][1] for args in junction) == ["f1", "f2"]
    assert all(args[1][0] == "model-1" for args in junction)


def test_ddl_creates_model_tables() -> None:
    assert "CREATE TABLE IF NOT EXISTS model_runs" in artifacts.DDL
    assert "CREATE TABLE IF NOT EXISTS model_run_features" in artifacts.JUNCTION_DDL
    assert "REFERENCES feature_runs(run_id)" in artifacts.JUNCTION_DDL


def test_fetch_latest_feature_runs() -> None:
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("uuid-1",),
        ("uuid-2",),
    ]
    ids = artifacts.fetch_latest_feature_runs(conn, ["a", "b"])
    assert ids == ["uuid-1", "uuid-2"]
    sql = conn.cursor.return_value.__enter__.return_value.execute.call_args.args[0]
    assert "DISTINCT ON (feature_group)" in sql
