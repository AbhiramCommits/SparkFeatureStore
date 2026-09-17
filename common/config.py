"""Configuration loading for sparkfeaturestore.

Config lives in conf/<env>.yaml. The active environment is selected with the
APP_ENV environment variable or an explicit argument (default: "local").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
DEFAULT_ENV = "local"


def load_config(env: str | None = None) -> dict[str, Any]:
    """Load conf/<env>.yaml and return it as a dict."""
    env = env or os.getenv("APP_ENV") or DEFAULT_ENV
    conf_path = CONF_DIR / f"{env}.yaml"
    if not conf_path.exists():
        available = ", ".join(p.stem for p in sorted(CONF_DIR.glob("*.yaml")))
        raise FileNotFoundError(
            f"Config for environment '{env}' not found at {conf_path} " f"(available: {available})"
        )
    with conf_path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config at {conf_path}: expected a YAML mapping")
    cfg.setdefault("environment", env)
    return cfg


def resolve_path(path: str | Path, env: str | None = None) -> str:
    """Resolve a config path to an absolute path.

    URLs (s3a://...) are returned unchanged. Relative paths are resolved
    against the repo root, so they work both on the host and inside
    containers where the repo is mounted at /opt/sparkfeaturestore.
    """
    path_str = str(path)
    if "://" in path_str:
        return path_str
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)
