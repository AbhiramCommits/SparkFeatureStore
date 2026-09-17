"""Gradient-boosted regression baseline for tip_pct.

Time-based train/val/test split (never random), a sklearn Pipeline with
imputation + one-hot encoding for the categorical zone/vendor ids, and
RMSE/MAE/R2 reported against a predict-the-mean baseline so the numbers
are interpretable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from common.config import load_config, resolve_path  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402
from features import registry as feature_registry  # noqa: E402
from training import artifacts, data  # noqa: E402

log = get_logger(__name__)

FRAMEWORK = "sklearn"
FEATURE_GROUPS = ["pickup_zone_demand", "driver_trip_history", "trip_features"]

HYPERPARAMS = {
    "model": "HistGradientBoostingRegressor",
    "max_iter": 400,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "random_state": 42,
}


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                data.NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore", min_frequency=20, sparse_output=False
                            ),
                        ),
                    ]
                ),
                data.CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = HistGradientBoostingRegressor(
        random_state=42,
        **{k: v for k, v in HYPERPARAMS.items() if k not in ("model", "random_state")},
    )
    return Pipeline([("prep", preprocessor), ("reg", model)])


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="local", help="Environment config (conf/<env>.yaml)")
    parser.add_argument("--run-id", default=None, help="Override artifact run id")
    parser.add_argument("--no-register", action="store_true", help="Skip Postgres model_runs")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    table_path = resolve_path(f"{cfg['paths']['silver']}/trip_features")
    frames, counts = data.load_training_data(table_path)

    pipeline = build_pipeline()
    log.info("Fitting %s on %d train rows", HYPERPARAMS["model"], counts["train"])
    pipeline.fit(
        frames["train"][data.FEATURE_COLUMNS],
        frames["train"][data.LABEL_COL],
    )

    baseline = float(frames["train"][data.LABEL_COL].mean())
    metrics: dict = {
        "framework": FRAMEWORK,
        "baseline": {
            "rmse": float(
                root_mean_squared_error(
                    frames["test"][data.LABEL_COL],
                    np.full(len(frames["test"]), baseline),
                )
            ),
            "mae": float(
                mean_absolute_error(
                    frames["test"][data.LABEL_COL], np.full(len(frames["test"]), baseline)
                )
            ),
        },
        "row_counts": counts,
    }
    for split in ("val", "test"):
        preds = pipeline.predict(frames[split][data.FEATURE_COLUMNS])
        metrics[split] = metrics_for(frames[split][data.LABEL_COL].to_numpy(), preds)

    run = artifacts.ArtifactRun(FRAMEWORK, run_id=args.run_id).ensure()
    artifacts.save_model(run, pipeline, FRAMEWORK)
    run.write_json("metrics", metrics)
    run.write_json("features", data.FEATURE_COLUMNS)
    run.write_json("hyperparams", HYPERPARAMS)

    git_sha = feature_registry.current_git_sha() or "uncommitted"
    conn = None
    feature_run_ids: list[str] = []
    if not args.no_register:
        conn = artifacts.connect(env=args.env)
        artifacts.ensure_table(conn)
        feature_run_ids = artifacts.fetch_latest_feature_runs(conn, FEATURE_GROUPS)
        artifacts.register_model_run(
            conn,
            run=run,
            framework=FRAMEWORK,
            metrics=metrics,
            features=data.FEATURE_COLUMNS,
            hyperparams=HYPERPARAMS,
            git_sha=git_sha,
            feature_run_ids=feature_run_ids,
        )
        conn.close()
    artifacts.write_metadata(
        run,
        features=data.FEATURE_COLUMNS,
        hyperparams=HYPERPARAMS,
        git_sha=git_sha,
        feature_run_ids=feature_run_ids,
        row_counts=counts,
    )

    log.info("=" * 64)
    log.info("sklearn training complete: run_id=%s", run.run_id)
    log.info("artifacts: %s", run.path)
    log.info("row counts: train=%d val=%d test=%d", counts["train"], counts["val"], counts["test"])
    log.info(
        "test:  rmse=%.4f mae=%.4f r2=%.4f",
        metrics["test"]["rmse"],
        metrics["test"]["mae"],
        metrics["test"]["r2"],
    )
    log.info(
        "baseline (mean): rmse=%.4f mae=%.4f",
        metrics["baseline"]["rmse"],
        metrics["baseline"]["mae"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
