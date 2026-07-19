"""Metrics compatible with B1, B2, and GraphSAGE."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, risk: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, risk)
    if len(thresholds) == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    y_true: np.ndarray, risk: np.ndarray, threshold: float
) -> dict[str, float | dict[str, int]]:
    prediction = (risk >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, float | dict[str, int]] = {
        "threshold": float(threshold),
        "f1_incompatible": float(f1_score(y_true, prediction, pos_label=1)),
        "precision_incompatible": float(
            precision_score(y_true, prediction, pos_label=1, zero_division=0)
        ),
        "recall_incompatible": float(
            recall_score(y_true, prediction, pos_label=1, zero_division=0)
        ),
        "pr_auc": float(average_precision_score(y_true, risk)),
        "roc_auc": float(roc_auc_score(y_true, risk)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    order = np.argsort(-risk, kind="stable")
    for k in (10, 50):
        metrics[f"precision_at_{k}"] = float(np.mean(y_true[order[: min(k, len(y_true))]]))
    return metrics
