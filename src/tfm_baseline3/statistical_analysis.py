"""Reproducible paired analysis over existing test predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_BINS = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
DESCRIPTIVE_METRICS = (
    "f1_incompatible",
    "precision_incompatible",
    "recall_incompatible",
    "accuracy",
    "balanced_accuracy",
    "pr_auc",
    "roc_auc",
    "precision_at_10",
    "precision_at_50",
    "brier",
    "log_loss",
    "ece",
)
DEFAULT_COMPARISON_METRICS = (
    "f1_incompatible",
    "precision_at_10",
    "precision_at_50",
    "brier",
    "log_loss",
    "ece",
)
METRIC_DIRECTION = {
    metric: "lower_is_better" if metric in {"brier", "log_loss", "ece"} else "higher_is_better"
    for metric in DESCRIPTIVE_METRICS
}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    root: Path
    summary: Path
    prediction_pattern: str
    pair_id_column: str = "pair_id"
    label_column: str = "label"
    risk_column: str = "risk_incompatible"
    prediction_column: str = "prediction"


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    name: str
    left: str
    right: str


@dataclass(slots=True)
class ExperimentData:
    spec: ExperimentSpec
    dataset_sha256: str
    seeds: list[int]
    predictions: dict[int, pd.DataFrame]
    summary_sha256: str
    prediction_sha256: dict[int, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse paired test predictions without retraining models"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
        help="Number of equal-width calibration bins on [0, 1]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")
    if args.ece_bins < 2:
        raise SystemExit("--ece-bins must be at least 2")
    config, experiments, comparisons = load_config(args.config)
    bins = np.linspace(0.0, 1.0, args.ece_bins + 1, dtype=float)
    report, tables = analyse(
        experiments,
        comparisons,
        comparison_metrics=tuple(
            config.get("comparison_metrics", DEFAULT_COMPARISON_METRICS)
        ),
        bins=bins,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "analysis.json"),
                "dataset_sha256": report["dataset_sha256"],
                "experiments": len(report["experiments"]),
                "comparisons": len(report["comparisons"]),
                "seeds": report["seeds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_config(
    path: Path,
) -> tuple[dict[str, Any], list[ExperimentSpec], list[ComparisonSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    experiments = []
    for item in payload.get("experiments", []):
        root = _resolve_path(base, item["root"])
        columns = item.get("columns", {})
        summary = _resolve_path(root, item.get("summary", "summary.json"))
        experiments.append(
            ExperimentSpec(
                name=item["name"],
                root=root,
                summary=summary,
                prediction_pattern=item["prediction_pattern"],
                pair_id_column=columns.get("pair_id", "pair_id"),
                label_column=columns.get("label", "label"),
                risk_column=columns.get("risk", "risk_incompatible"),
                prediction_column=columns.get("prediction", "prediction"),
            )
        )
    comparisons = [ComparisonSpec(**item) for item in payload.get("comparisons", [])]
    _validate_specs(experiments, comparisons)
    return payload, experiments, comparisons


def analyse(
    experiment_specs: list[ExperimentSpec],
    comparisons: list[ComparisonSpec],
    *,
    comparison_metrics: tuple[str, ...] = DEFAULT_COMPARISON_METRICS,
    bins: np.ndarray | Sequence[float] = DEFAULT_BINS,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    bin_edges = _validate_bins(np.asarray(bins, dtype=float))
    unknown_metrics = set(comparison_metrics) - set(DESCRIPTIVE_METRICS)
    if unknown_metrics:
        raise ValueError(f"Unknown comparison metrics: {sorted(unknown_metrics)}")
    data = {spec.name: _load_experiment(spec) for spec in experiment_specs}
    dataset_sha256, seeds, pair_ids = _validate_compatible(data)

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    experiment_reports: dict[str, Any] = {}
    metric_values: dict[str, dict[int, dict[str, float]]] = {}
    for name, experiment in data.items():
        metric_values[name] = {}
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            predictions = experiment.predictions[seed]
            metrics, calibration = prediction_metrics(predictions, bin_edges)
            metric_values[name][seed] = metrics
            per_seed[str(seed)] = {"metrics": metrics, "calibration": calibration}
            metric_rows.extend(
                {
                    "experiment": name,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                    "direction": METRIC_DIRECTION[metric],
                }
                for metric, value in metrics.items()
            )
            calibration_rows.extend(
                {"experiment": name, "seed": seed, **row} for row in calibration
            )
        experiment_reports[name] = {
            "root": str(experiment.spec.root),
            "summary": str(experiment.spec.summary),
            "summary_sha256": experiment.summary_sha256,
            "prediction_pattern": experiment.spec.prediction_pattern,
            "prediction_sha256": {
                str(seed): experiment.prediction_sha256[seed] for seed in seeds
            },
            "aggregate": {
                metric: _summary([metric_values[name][seed][metric] for seed in seeds])
                for metric in DESCRIPTIVE_METRICS
            },
            "per_seed": per_seed,
        }

    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    comparison_reports: dict[str, Any] = {}
    hypotheses: list[tuple[str, str, float]] = []
    bootstrap_cache = {
        (name, seed): _bootstrap_distributions(
            experiment.predictions[seed],
            metrics=comparison_metrics,
            bins=bin_edges,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + seed,
        )
        for name, experiment in data.items()
        for seed in seeds
    }
    for comparison in comparisons:
        comparison_report: dict[str, Any] = {
            "left": comparison.left,
            "right": comparison.right,
            "delta_definition": "right_minus_left",
            "metrics": {},
            "paired_bootstrap_by_seed": {},
        }
        for metric in comparison_metrics:
            deltas = np.asarray(
                [
                    metric_values[comparison.right][seed][metric]
                    - metric_values[comparison.left][seed][metric]
                    for seed in seeds
                ],
                dtype=float,
            )
            test = _wilcoxon(deltas)
            comparison_report["metrics"][metric] = {
                "direction": METRIC_DIRECTION[metric],
                "delta": _delta_summary(deltas, seeds, METRIC_DIRECTION[metric]),
                "wilcoxon_two_sided": test,
            }
            hypotheses.append((comparison.name, metric, test["pvalue"]))
            for seed, delta in zip(seeds, deltas):
                comparison_rows.append(
                    {
                        "comparison": comparison.name,
                        "left": comparison.left,
                        "right": comparison.right,
                        "seed": seed,
                        "metric": metric,
                        "direction": METRIC_DIRECTION[metric],
                        "left_value": metric_values[comparison.left][seed][metric],
                        "right_value": metric_values[comparison.right][seed][metric],
                        "delta_right_minus_left": float(delta),
                    }
                )
        for seed in seeds:
            bootstrap = _summarize_paired_bootstrap(
                metric_values[comparison.left][seed],
                metric_values[comparison.right][seed],
                bootstrap_cache[(comparison.left, seed)],
                bootstrap_cache[(comparison.right, seed)],
                metrics=comparison_metrics,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + seed,
            )
            comparison_report["paired_bootstrap_by_seed"][str(seed)] = bootstrap
            bootstrap_rows.extend(
                {
                    "comparison": comparison.name,
                    "left": comparison.left,
                    "right": comparison.right,
                    "seed": seed,
                    "metric": metric,
                    "direction": METRIC_DIRECTION[metric],
                    "estimate_delta_right_minus_left": values["estimate"],
                    "bootstrap_mean_delta": values["bootstrap_mean"],
                    "ci_95_lower": values["confidence_interval_95"][0],
                    "ci_95_upper": values["confidence_interval_95"][1],
                    "resamples": bootstrap_resamples,
                    "bootstrap_seed": bootstrap_seed + seed,
                }
                for metric, values in bootstrap["metrics"].items()
            )
        comparison_reports[comparison.name] = comparison_report

    adjusted = holm_adjust([pvalue for _, _, pvalue in hypotheses])
    wilcoxon_rows = []
    for (comparison_name, metric, raw_pvalue), adjusted_pvalue in zip(hypotheses, adjusted):
        comparison_reports[comparison_name]["metrics"][metric]["wilcoxon_two_sided"][
            "holm_adjusted_pvalue"
        ] = adjusted_pvalue
        wilcoxon_rows.append(
            {
                "comparison": comparison_name,
                "metric": metric,
                "direction": METRIC_DIRECTION[metric],
                "raw_pvalue": raw_pvalue,
                "holm_adjusted_pvalue": adjusted_pvalue,
                "family_size": len(hypotheses),
            }
        )

    report = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "seeds": seeds,
        "test_pairs": len(pair_ids),
        "methods": {
            "precision_at_k": {
                "k": [10, 50],
                "ranking": "risk descending; pair_id ascending breaks ties",
            },
            "calibration": {
                "brier": "mean squared error between risk and label",
                "log_loss_probability_clip": [1e-15, 1.0 - 1e-15],
                "ece": "sum over bins of bin_fraction * abs(mean_risk - positive_rate)",
                "bin_edges": bin_edges.tolist(),
                "bin_convention": "[lower, upper), with the final bin including 1.0",
            },
            "paired_bootstrap": {
                "unit": "test pair",
                "resamples": bootstrap_resamples,
                "base_seed": bootstrap_seed,
                "seed_for_run": "base_seed + training_seed",
                "interval": "percentile 95%",
                "pairing": "same resampled pair indices for left and right",
            },
            "paired_seed_test": {
                "test": "two-sided Wilcoxon signed-rank over right-minus-left seed deltas",
                "zero_method": "wilcox: zero differences are excluded from the ranks",
                "method_selection": "exact without zeros; normal approximation with zeros",
                "continuity_correction": "applied only with the normal approximation",
                "multiple_testing": "Holm step-down over all comparison-metric hypotheses",
                "family_size": len(hypotheses),
            },
        },
        "comparison_metrics": list(comparison_metrics),
        "experiments": experiment_reports,
        "comparisons": comparison_reports,
    }
    tables = {
        "per_seed_metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "paired_comparisons": pd.DataFrame(comparison_rows),
        "paired_bootstrap": pd.DataFrame(bootstrap_rows),
        "wilcoxon_holm": pd.DataFrame(wilcoxon_rows),
    }
    return report, tables


def prediction_metrics(
    predictions: pd.DataFrame,
    bins: np.ndarray | Sequence[float] = DEFAULT_BINS,
    metrics: Sequence[str] = DESCRIPTIVE_METRICS,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    requested = set(metrics)
    unknown = requested - set(DESCRIPTIVE_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    y_true = predictions["label"].to_numpy(dtype="int8")
    risk = predictions["risk"].to_numpy(dtype="float64")
    predicted = predictions["prediction"].to_numpy(dtype="int8")
    pair_ids = predictions["pair_id"].astype(str).to_numpy()
    calibration = calibration_table(y_true, risk, bins)
    clipped = np.clip(risk, 1e-15, 1.0 - 1e-15)
    candidates = {
        "f1_incompatible": lambda: float(f1_score(y_true, predicted, pos_label=1)),
        "precision_incompatible": lambda: float(
            precision_score(y_true, predicted, pos_label=1, zero_division=0)
        ),
        "recall_incompatible": lambda: float(
            recall_score(y_true, predicted, pos_label=1, zero_division=0)
        ),
        "accuracy": lambda: float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": lambda: float(balanced_accuracy_score(y_true, predicted)),
        "pr_auc": lambda: float(average_precision_score(y_true, risk)),
        "roc_auc": lambda: float(roc_auc_score(y_true, risk)),
        "precision_at_10": lambda: _precision_at_k(y_true, risk, pair_ids, 10),
        "precision_at_50": lambda: _precision_at_k(y_true, risk, pair_ids, 50),
        "brier": lambda: float(np.mean(np.square(risk - y_true))),
        "log_loss": lambda: float(
            -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
        ),
        "ece": lambda: float(sum(row["ece_contribution"] for row in calibration)),
    }
    return {metric: candidates[metric]() for metric in metrics}, calibration


def calibration_table(
    y_true: np.ndarray,
    risk: np.ndarray,
    bins: np.ndarray | Sequence[float] = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    edges = _validate_bins(np.asarray(bins, dtype=float))
    assignments = np.searchsorted(edges[1:-1], risk, side="right")
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = assignments == index
        count = int(selected.sum())
        mean_risk = float(risk[selected].mean()) if count else None
        positive_rate = float(y_true[selected].mean()) if count else None
        absolute_gap = abs(mean_risk - positive_rate) if count else None
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "fraction": count / len(y_true),
                "mean_risk": mean_risk,
                "positive_rate": positive_rate,
                "absolute_gap": absolute_gap,
                "ece_contribution": (count / len(y_true)) * absolute_gap if count else 0.0,
            }
        )
    return rows


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metrics: Sequence[str],
    bins: np.ndarray | Sequence[float] = DEFAULT_BINS,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    _validate_prediction_pair(left, right)
    left_metrics, _ = prediction_metrics(left, bins, metrics)
    right_metrics, _ = prediction_metrics(right, bins, metrics)
    left_distributions = _bootstrap_distributions(
        left, metrics=metrics, bins=bins, resamples=resamples, seed=seed
    )
    right_distributions = _bootstrap_distributions(
        right, metrics=metrics, bins=bins, resamples=resamples, seed=seed
    )
    return _summarize_paired_bootstrap(
        left_metrics,
        right_metrics,
        left_distributions,
        right_distributions,
        metrics=metrics,
        resamples=resamples,
        seed=seed,
    )


def _bootstrap_distributions(
    predictions: pd.DataFrame,
    *,
    metrics: Sequence[str],
    bins: np.ndarray | Sequence[float],
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    distributions = {metric: np.empty(resamples, dtype=float) for metric in metrics}
    rng = np.random.default_rng(seed)
    for iteration in range(resamples):
        indexes = rng.integers(0, len(predictions), size=len(predictions))
        sample_metrics, _ = prediction_metrics(predictions.iloc[indexes], bins, metrics)
        for metric in metrics:
            distributions[metric][iteration] = sample_metrics[metric]
    return distributions


def _summarize_paired_bootstrap(
    left_metrics: dict[str, float],
    right_metrics: dict[str, float],
    left_distributions: dict[str, np.ndarray],
    right_distributions: dict[str, np.ndarray],
    *,
    metrics: Sequence[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    delta_distributions = {
        metric: right_distributions[metric] - left_distributions[metric] for metric in metrics
    }
    return {
        "resamples": resamples,
        "seed": seed,
        "metrics": {
            metric: {
                "estimate": right_metrics[metric] - left_metrics[metric],
                "bootstrap_mean": float(distribution.mean()),
                "confidence_interval_95": [
                    float(np.quantile(distribution, 0.025)),
                    float(np.quantile(distribution, 0.975)),
                ],
            }
            for metric, distribution in delta_distributions.items()
        },
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _load_experiment(spec: ExperimentSpec) -> ExperimentData:
    summary = json.loads(spec.summary.read_text(encoding="utf-8"))
    dataset_sha256 = str(summary["dataset"]["sha256"])
    protocol = summary.get("protocol")
    raw_seeds = protocol.get("seeds") if isinstance(protocol, dict) else summary.get("seeds")
    if raw_seeds is None:
        raise ValueError(f"Experiment {spec.name} summary does not declare seeds")
    seeds = [int(seed) for seed in raw_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Experiment {spec.name} has invalid seeds")
    predictions = {}
    prediction_sha256 = {}
    for seed in seeds:
        path = spec.root / spec.prediction_pattern.format(seed=seed)
        if not path.is_file():
            raise FileNotFoundError(f"Missing predictions for {spec.name} seed {seed}: {path}")
        frame = pd.read_parquet(path)
        rename = {
            spec.pair_id_column: "pair_id",
            spec.label_column: "label",
            spec.risk_column: "risk",
            spec.prediction_column: "prediction",
        }
        missing = set(rename) - set(frame.columns)
        if missing:
            raise ValueError(f"Predictions {path} are missing columns: {sorted(missing)}")
        canonical = frame.rename(columns=rename)[list(rename.values())].copy()
        _validate_predictions(canonical, spec.name, seed)
        predictions[seed] = canonical.sort_values("pair_id", kind="stable").reset_index(drop=True)
        prediction_sha256[seed] = _sha256(path)
    return ExperimentData(
        spec,
        dataset_sha256,
        seeds,
        predictions,
        _sha256(spec.summary),
        prediction_sha256,
    )


def _validate_predictions(frame: pd.DataFrame, name: str, seed: int) -> None:
    if frame["pair_id"].isna().any() or frame["pair_id"].duplicated().any():
        raise ValueError(f"Experiment {name} seed {seed} has null or duplicate pair_id values")
    labels = frame["label"].to_numpy()
    predictions = frame["prediction"].to_numpy()
    risk = frame["risk"].to_numpy(dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Experiment {name} seed {seed} must contain both binary labels")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError(f"Experiment {name} seed {seed} has non-binary predictions")
    if np.any(~np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
        raise ValueError(f"Experiment {name} seed {seed} has invalid risks")


def _validate_compatible(
    experiments: dict[str, ExperimentData],
) -> tuple[str, list[int], np.ndarray]:
    reference = next(iter(experiments.values()))
    reference_seed = reference.seeds[0]
    reference_frame = reference.predictions[reference_seed]
    pair_ids = reference_frame["pair_id"].astype(str).to_numpy()
    labels = reference_frame["label"].to_numpy(dtype="int8")
    for experiment in experiments.values():
        if experiment.dataset_sha256 != reference.dataset_sha256:
            raise ValueError("Experiments use different dataset hashes")
        if experiment.seeds != reference.seeds:
            raise ValueError("Experiments use different ordered seed lists")
        for seed, frame in experiment.predictions.items():
            if not np.array_equal(frame["pair_id"].astype(str).to_numpy(), pair_ids):
                raise ValueError(f"Experiment {experiment.spec.name} seed {seed} has different pair_ids")
            if not np.array_equal(frame["label"].to_numpy(dtype="int8"), labels):
                raise ValueError(f"Experiment {experiment.spec.name} seed {seed} has different labels")
    return reference.dataset_sha256, reference.seeds, pair_ids


def _validate_prediction_pair(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in ("pair_id", "label"):
        if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
            raise ValueError(f"Paired predictions differ in {column}")


def _precision_at_k(
    y_true: np.ndarray, risk: np.ndarray, pair_ids: np.ndarray, k: int
) -> float:
    order = np.lexsort((pair_ids, -risk))
    return float(np.mean(y_true[order[: min(k, len(y_true))]]))


def _wilcoxon(deltas: np.ndarray) -> dict[str, float | str | bool]:
    if np.allclose(deltas, 0.0, atol=1e-15):
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "method": "all_deltas_zero",
            "zero_method": "wilcox",
            "correction": False,
        }
    has_zero = bool(np.any(np.isclose(deltas, 0.0, atol=1e-15)))
    method = "approx" if has_zero else "exact"
    result = wilcoxon(
        deltas,
        alternative="two-sided",
        zero_method="wilcox",
        correction=has_zero,
        method=method,
    )
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "method": method,
        "zero_method": "wilcox",
        "correction": has_zero,
    }


def _delta_summary(
    deltas: np.ndarray, seeds: list[int], direction: str
) -> dict[str, Any]:
    summary = _summary(deltas.tolist())
    summary["per_seed"] = {str(seed): float(delta) for seed, delta in zip(seeds, deltas)}
    right_better = deltas > 0 if direction == "higher_is_better" else deltas < 0
    left_better = deltas < 0 if direction == "higher_is_better" else deltas > 0
    summary["right_better"] = int(np.sum(right_better))
    summary["ties"] = int(np.sum(np.isclose(deltas, 0.0, atol=1e-15)))
    summary["left_better"] = int(np.sum(left_better))
    return summary


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    mean = float(array.mean())
    if len(array) > 1 and not np.isclose(std, 0.0, atol=1e-15):
        interval = t.interval(0.95, len(array) - 1, loc=mean, scale=std / np.sqrt(len(array)))
        confidence = [float(interval[0]), float(interval[1])]
    else:
        confidence = [mean, mean]
    return {
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "confidence_interval_95": confidence,
    }


def _validate_bins(edges: np.ndarray) -> np.ndarray:
    if (
        edges.ndim != 1
        or len(edges) < 3
        or not np.isclose(edges[0], 0.0)
        or not np.isclose(edges[-1], 1.0)
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("Calibration bin edges must increase strictly from 0 to 1")
    return edges


def _validate_specs(
    experiments: list[ExperimentSpec], comparisons: list[ComparisonSpec]
) -> None:
    names = [experiment.name for experiment in experiments]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("At least two uniquely named experiments are required")
    known = set(names)
    comparison_names = [comparison.name for comparison in comparisons]
    if not comparisons or len(comparison_names) != len(set(comparison_names)):
        raise ValueError("At least one uniquely named comparison is required")
    for comparison in comparisons:
        if comparison.left not in known or comparison.right not in known:
            raise ValueError(f"Comparison {comparison.name} references an unknown experiment")
        if comparison.left == comparison.right:
            raise ValueError(f"Comparison {comparison.name} must use two experiments")


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
