"""Training and artifacts for the neural edge-only B3 baseline."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from packaging import __version__ as packaging_version
from torch import nn

from tfm_baseline3.evaluation import choose_threshold, classification_metrics
from tfm_baseline3.features import (
    CATEGORICAL_FEATURES,
    build_transformer,
    numeric_feature_names,
    raw_edge_features,
)
from tfm_baseline3.model import EdgeMLPClassifier
from tfm_baseline3.provenance import runtime_environment, synchronize
from tfm_baseline3.schema import SPLITS, validate_dataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    hidden_channels: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    temporal_batch_size: int = 256
    device: str = "auto"
    include_vb_in_range: bool = True


def train_experiment(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    config: TrainingConfig | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()
    _validate_config(config)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    data = validate_dataset(frame)
    positions = {
        split: np.flatnonzero(data["split"].eq(split).to_numpy()).tolist() for split in SPLITS
    }
    numeric_features = numeric_feature_names(
        include_vb_in_range=config.include_vb_in_range
    )
    raw_features = raw_edge_features(
        data, include_vb_in_range=config.include_vb_in_range
    )
    transformer = build_transformer(
        include_vb_in_range=config.include_vb_in_range
    )
    transformer.fit(raw_features.iloc[positions["train"]])
    feature_values = transformer.transform(raw_features).astype("float32")
    features = torch.from_numpy(feature_values).to(device)
    labels = torch.tensor(data["label"].to_numpy(), dtype=torch.float32, device=device)
    model = EdgeMLPClassifier(
        features.shape[1], hidden_channels=config.hidden_channels, dropout=config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_labels = data.iloc[positions["train"]]["label"].to_numpy()
    positives = int(np.sum(train_labels == 1))
    negatives = int(np.sum(train_labels == 0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -1.0
    stale_epochs = 0
    fit_started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(
            model,
            optimizer,
            criterion,
            features,
            labels,
            data,
            positions["train"],
            config.temporal_batch_size,
            device,
        )
        validation_risk, _ = _predict(
            model,
            features,
            data,
            positions["val"],
            config.temporal_batch_size,
            device,
        )
        validation_labels = data.iloc[positions["val"]]["label"].to_numpy(dtype="int8")
        threshold = choose_threshold(validation_labels, validation_risk)
        validation_metrics = classification_metrics(validation_labels, validation_risk, threshold)
        score = float(validation_metrics["f1_incompatible"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_f1_incompatible": score,
                "validation_pr_auc": float(validation_metrics["pr_auc"]),
                "threshold": threshold,
            }
        )
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    fit_seconds = time.perf_counter() - fit_started
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)

    validation_risk, validation_seconds = _predict(
        model,
        features,
        data,
        positions["val"],
        config.temporal_batch_size,
        device,
    )
    validation_labels = data.iloc[positions["val"]]["label"].to_numpy(dtype="int8")
    threshold = choose_threshold(validation_labels, validation_risk)
    test_risk, test_seconds = _predict(
        model,
        features,
        data,
        positions["test"],
        config.temporal_batch_size,
        device,
    )
    test_labels = data.iloc[positions["test"]]["label"].to_numpy(dtype="int8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    checkpoint_path = output_dir / "models/baseline3_mlp.pt"
    preprocessor_path = output_dir / "models/preprocessor.joblib"
    prediction_path = output_dir / "predictions/baseline3_test.parquet"
    torch.save(
        {
            "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
            "config": asdict(config),
            "feature_count": int(features.shape[1]),
            "best_epoch": best_epoch,
            "threshold": threshold,
        },
        checkpoint_path,
    )
    joblib.dump(transformer, preprocessor_path)
    _write_predictions(data.iloc[positions["test"]], test_risk, threshold, prediction_path)

    report: dict[str, Any] = {
        "baseline": "B3",
        "model": "EdgeMLP",
        "config": asdict(config),
        "dataset": dataset_metadata or {"rows": int(len(data))},
        "environment": runtime_environment(
            device=device,
            libraries={
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "packaging": packaging_version,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        ),
        "features": {
            "numeric": numeric_features,
            "categorical": CATEGORICAL_FEATURES,
            "vb_in_range_included": config.include_vb_in_range,
            "package_identity_excluded": True,
            "uses_graph": False,
        },
        "rows": {split: len(positions[split]) for split in SPLITS},
        "training": {
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "fit_seconds": fit_seconds,
            "history": history,
        },
        "validation": {
            **classification_metrics(validation_labels, validation_risk, threshold),
            "inference": {
                "total_seconds": validation_seconds,
                "seconds_per_pair": validation_seconds / len(validation_labels),
            },
        },
        "test": {
            **classification_metrics(test_labels, test_risk, threshold),
            "inference": {
                "total_seconds": test_seconds,
                "seconds_per_pair": test_seconds / len(test_labels),
            },
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "preprocessor": str(preprocessor_path),
            "test_predictions": str(prediction_path),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _train_epoch(
    model: EdgeMLPClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    data: pd.DataFrame,
    positions: list[int],
    batch_size: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in _temporal_batches(data, positions, batch_size):
        index = torch.tensor(batch, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features[index]), labels[index])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
    return total_loss / len(positions)


def _predict(
    model: EdgeMLPClassifier,
    features: torch.Tensor,
    data: pd.DataFrame,
    positions: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    risks: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in _temporal_batches(data, positions, batch_size):
            index = torch.tensor(batch, dtype=torch.long, device=device)
            risks.append(torch.sigmoid(model(features[index])).cpu().numpy())
    synchronize(device)
    return np.concatenate(risks), time.perf_counter() - started


def _temporal_batches(data: pd.DataFrame, positions: list[int], size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_time: pd.Timestamp | None = None
    for position in positions:
        event_time = data.iloc[position]["event_time"]
        if current and event_time != current_time and len(current) >= size:
            batches.append(current)
            current = []
        current.append(position)
        current_time = event_time
    if current:
        batches.append(current)
    return batches


def _write_predictions(rows: pd.DataFrame, risk: np.ndarray, threshold: float, path: Path) -> None:
    columns = ["pair_id", "pa", "va", "pb", "vb", "declared_range", "split", "label"]
    predictions = rows[columns].copy()
    predictions["risk_incompatible"] = risk
    predictions["prediction"] = (risk >= threshold).astype("int8")
    predictions.to_parquet(path, index=False)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _validate_config(config: TrainingConfig) -> None:
    if config.max_epochs < 1 or config.patience < 1 or config.temporal_batch_size < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
