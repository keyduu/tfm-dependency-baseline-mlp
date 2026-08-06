"""Repeated inference-only benchmark for saved B3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from packaging import __version__ as packaging_version

from tfm_baseline3.features import raw_edge_features
from tfm_baseline3.model import EdgeMLPClassifier
from tfm_baseline3.provenance import runtime_environment
from tfm_baseline3.schema import validate_dataset
from tfm_baseline3.training import _predict


def run_inference_benchmark(
    frame: pd.DataFrame,
    dataset_path: Path,
    benchmark_root: Path,
    output_path: Path,
    *,
    device_name: str = "auto",
    warmup_runs: int = 3,
    measured_runs: int = 10,
) -> dict[str, Any]:
    """Benchmark every saved B3 model without performing any fitting."""

    _validate_repetitions(warmup_runs, measured_runs)
    device = _resolve_device(device_name)
    data = validate_dataset(frame)
    test_positions = np.flatnonzero(data["split"].eq("test").to_numpy()).tolist()
    checkpoint_paths = sorted(benchmark_root.glob("**/models/baseline3_mlp.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No saved B3 checkpoints found below {benchmark_root}")

    runs = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for checkpoint_path in checkpoint_paths:
        run_dir = checkpoint_path.parent.parent
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        include_vb = bool(config.get("include_vb_in_range", True))
        transformer = joblib.load(run_dir / "models/preprocessor.joblib")
        raw_features = raw_edge_features(data, include_vb_in_range=include_vb)
        feature_values = transformer.transform(raw_features).astype("float32")
        features = torch.from_numpy(feature_values).to(device)
        model = EdgeMLPClassifier(
            int(checkpoint["feature_count"]),
            hidden_channels=int(config["hidden_channels"]),
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        reference, _ = _predict(
            model,
            features,
            data,
            test_positions,
            int(config["temporal_batch_size"]),
            device,
        )
        _check_saved_predictions(run_dir, reference)
        for _ in range(warmup_runs):
            risk, _ = _predict(
                model,
                features,
                data,
                test_positions,
                int(config["temporal_batch_size"]),
                device,
            )
            np.testing.assert_allclose(risk, reference, rtol=1e-5, atol=1e-6)
        timings = []
        for _ in range(measured_runs):
            risk, seconds = _predict(
                model,
                features,
                data,
                test_positions,
                int(config["temporal_batch_size"]),
                device,
            )
            timings.append(seconds)
            np.testing.assert_allclose(risk, reference, rtol=1e-5, atol=1e-6)

        protocol = "with_vb_in_range" if include_vb else "without_vb_in_range"
        grouped[protocol].extend(timings)
        runs.append(
            {
                "model": "B3 EdgeMLP",
                "protocol": protocol,
                "seed": _seed_from_path(checkpoint_path, config),
                "artifact": str(checkpoint_path),
                "predictions_verified": True,
                "total_seconds": timings,
                **_summary(timings, len(test_positions)),
            }
        )

    report = {
        "benchmark": "B3 inference-only",
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "rows": len(data),
            "test_rows": len(test_positions),
        },
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
        "measurement": {
            "clock": "time.perf_counter with explicit CUDA synchronization",
            "warmup_runs_per_model": warmup_runs,
            "measured_runs_per_model": measured_runs,
            "scope": "test batching, tensor indexing, forward pass and CPU risk transfer",
            "excluded": [
                "Parquet I/O",
                "feature engineering and fitted transformation",
                "model loading",
                "process startup",
                "JSON serialization",
            ],
        },
        "runs": runs,
        "aggregate": {
            protocol: {
                "saved_models": sum(run["protocol"] == protocol for run in runs),
                **_summary(values, len(test_positions)),
            }
            for protocol, values in sorted(grouped.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _check_saved_predictions(run_dir: Path, reference: np.ndarray) -> None:
    path = run_dir / "predictions/baseline3_test.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Saved predictions not found: {path}")
    expected = pd.read_parquet(path, columns=["risk_incompatible"])[
        "risk_incompatible"
    ].to_numpy()
    np.testing.assert_allclose(reference, expected, rtol=1e-5, atol=1e-6)


def _seed_from_path(path: Path, config: dict[str, Any]) -> int | None:
    match = re.search(r"seed[_-](\d+)", str(path))
    return int(match.group(1)) if match else config.get("seed")


def _summary(values: list[float], rows: int) -> dict[str, float | int]:
    mean = fmean(values)
    return {
        "measured_runs": len(values),
        "mean_seconds": mean,
        "std_seconds": pstdev(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds_per_pair": mean / rows,
        "throughput_pairs_per_second": rows / mean,
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _validate_repetitions(warmup_runs: int, measured_runs: int) -> None:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark saved B3 inference without training")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/_local/processed/dataset_mvd_v2_2.parquet"),
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_inference_benchmark(
        pd.read_parquet(args.dataset),
        args.dataset,
        args.benchmark_root,
        args.output,
        device_name=args.device,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
