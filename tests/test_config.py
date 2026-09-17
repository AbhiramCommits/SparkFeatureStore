"""Tests for common.config."""

from __future__ import annotations

import pytest

from common.config import REPO_ROOT, load_config, resolve_path


def test_load_local_config() -> None:
    cfg = load_config("local")
    assert cfg["environment"] == "local"
    profiles = cfg["spark"]["profiles"]
    assert {"local", "cluster", "k8s"} <= set(profiles)
    assert profiles["local"]["master"] == "local[*]"
    assert profiles["cluster"]["master"] == "spark://spark-master:7077"
    assert profiles["k8s"]["master"].startswith("k8s://")
    assert cfg["s3a"]["endpoint"] == "http://localhost:9000"


def test_load_all_env_configs() -> None:
    for env in ("local", "docker", "k8s"):
        cfg = load_config(env)
        assert cfg["paths"]["bronze"]
        assert cfg["spark"]["profiles"]


def test_resolve_local_path() -> None:
    assert resolve_path("data/raw") == str(REPO_ROOT / "data" / "raw")


def test_resolve_path_is_idempotent_for_absolute() -> None:
    assert resolve_path("/data/raw") == "/data/raw"


def test_resolve_remote_path_unchanged() -> None:
    assert resolve_path("s3a://datalake/bronze/trips") == "s3a://datalake/bronze/trips"


def test_missing_env_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist")
