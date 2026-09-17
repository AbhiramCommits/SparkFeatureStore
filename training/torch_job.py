"""PyTorch MLP for tip_pct with embedding layers for categorical ids.

Same time-based split as the sklearn job (training/data.py), a proper
Dataset/DataLoader pipeline, early stopping on the val loss, and
GPU-optional execution (cuda when available, cpu otherwise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common.config import load_config, resolve_path  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402
from features import registry as feature_registry  # noqa: E402
from training import artifacts, data  # noqa: E402

log = get_logger(__name__)

FRAMEWORK = "torch"
FEATURE_GROUPS = ["pickup_zone_demand", "driver_trip_history", "trip_features"]

HYPERPARAMS = {
    "embedding_dim": 8,
    "hidden": [64, 32],
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "max_epochs": 30,
    "early_stopping_patience": 5,
    "seed": 42,
}


class TripsDataset(Dataset):
    def __init__(
        self,
        frame,
        numeric_cols: list[str],
        cat_cols: list[str],
        label_col: str,
        num_mean: np.ndarray | None = None,
        num_std: np.ndarray | None = None,
    ):
        self.frame = frame
        self.X_num = frame[numeric_cols].fillna(0.0).astype("float32").to_numpy()
        self.X_cat = frame[cat_cols].fillna(0).astype("int64").to_numpy()
        self.y = frame[label_col].astype("float32").to_numpy()
        self.num_mean = num_mean if num_mean is not None else self.X_num.mean(axis=0)
        self.num_std = num_std if num_std is not None else self.X_num.std(axis=0) + 1e-6
        self.cat_sizes = [int(self.X_cat[:, i].max()) + 1 for i in range(self.X_cat.shape[1])]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        num = (self.X_num[idx] - self.num_mean) / self.num_std
        cat = self.X_cat[idx].copy()
        for i, size in enumerate(self.cat_sizes):
            if cat[i] >= size or cat[i] < 0:
                cat[i] = 0  # unknown id -> embedding 0
        return (
            torch.tensor(num, dtype=torch.float32),
            torch.tensor(cat, dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class TipMLP(nn.Module):
    def __init__(self, n_numeric: int, cat_sizes: list[int], embedding_dim: int, hidden: list[int]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in cat_sizes])
        layers: list[nn.Module] = []
        in_dim = n_numeric + embedding_dim * len(cat_sizes)
        for out_dim in hidden:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num, *embeds], dim=1)
        return self.mlp(x).squeeze(-1)


def train(model, train_dl, val_dl, device, lr, max_epochs, patience) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_num, x_cat, y in train_dl:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_num, x_cat), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num, x_cat, y in val_dl:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                val_loss += loss_fn(model(x_num, x_cat), y).item() * len(y)
        val_loss /= len(val_dl.dataset)
        log.info("epoch %02d: val mse=%.5f", epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break
    assert best_state is not None, "no training progress"
    model.load_state_dict(best_state)
    return model


def predict(model, dl, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x_num, x_cat, _y in dl:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            preds.append(model(x_num, x_cat).cpu().numpy())
    return np.concatenate(preds)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="local", help="Environment config (conf/<env>.yaml)")
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["max_epochs"])
    parser.add_argument("--run-id", default=None, help="Override artifact run id")
    parser.add_argument("--no-register", action="store_true", help="Skip Postgres model_runs")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    cfg = load_config(args.env)
    table_path = resolve_path(f"{cfg['paths']['silver']}/trip_features")
    frames, counts = data.load_training_data(table_path)

    torch.manual_seed(HYPERPARAMS["seed"])
    train_ds = TripsDataset(
        frames["train"], data.NUMERIC_FEATURES, data.CATEGORICAL_FEATURES, data.LABEL_COL
    )
    val_ds = TripsDataset(
        frames["val"],
        data.NUMERIC_FEATURES,
        data.CATEGORICAL_FEATURES,
        data.LABEL_COL,
        num_mean=train_ds.num_mean,
        num_std=train_ds.num_std,
    )
    test_ds = TripsDataset(
        frames["test"],
        data.NUMERIC_FEATURES,
        data.CATEGORICAL_FEATURES,
        data.LABEL_COL,
        num_mean=train_ds.num_mean,
        num_std=train_ds.num_std,
    )
    train_dl = DataLoader(train_ds, batch_size=HYPERPARAMS["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=HYPERPARAMS["batch_size"])
    test_dl = DataLoader(test_ds, batch_size=HYPERPARAMS["batch_size"])

    model = TipMLP(
        n_numeric=len(data.NUMERIC_FEATURES),
        cat_sizes=train_ds.cat_sizes,
        embedding_dim=HYPERPARAMS["embedding_dim"],
        hidden=HYPERPARAMS["hidden"],
    ).to(device)
    model = train(
        model,
        train_dl,
        val_dl,
        device,
        lr=HYPERPARAMS["learning_rate"],
        max_epochs=args.epochs,
        patience=HYPERPARAMS["early_stopping_patience"],
    )

    y_test = test_ds.y
    preds = predict(model, test_dl, device)
    baseline = float(train_ds.y.mean())

    def rmse(y, p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    def mae(y, p):
        return float(np.mean(np.abs(y - p)))

    r2 = 1 - float(np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    metrics: dict = {
        "framework": FRAMEWORK,
        "test": {"rmse": rmse(y_test, preds), "mae": mae(y_test, preds), "r2": r2},
        "val": {
            "rmse": rmse(val_ds.y, predict(model, val_dl, device)),
            "mae": mae(val_ds.y, predict(model, val_dl, device)),
            "r2": 0.0,
        },
        "baseline": {
            "rmse": rmse(y_test, np.full_like(y_test, baseline)),
            "mae": mae(y_test, np.full_like(y_test, baseline)),
        },
        "row_counts": counts,
    }

    run = artifacts.ArtifactRun(FRAMEWORK, run_id=args.run_id).ensure()
    artifacts.save_model(run, model, FRAMEWORK)
    run.write_json("metrics", metrics)
    run.write_json("features", data.FEATURE_COLUMNS)
    run.write_json("hyperparams", {**HYPERPARAMS, "max_epochs": args.epochs, "device": device})

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
        hyperparams={**HYPERPARAMS, "device": device},
        git_sha=git_sha,
        feature_run_ids=feature_run_ids,
        row_counts=counts,
    )

    log.info("=" * 64)
    log.info("torch training complete: run_id=%s (device=%s)", run.run_id, device)
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
