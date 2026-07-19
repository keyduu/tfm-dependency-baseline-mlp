"""Ten-seed protocol for baseline B3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

DEFAULT_SEEDS = tuple(range(42, 52))
METRICS = (
    "f1_incompatible",
    "precision_incompatible",
    "recall_incompatible",
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ten seeds for baseline B3")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/_local/processed/dataset_mvd_v2.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/_local/baseline3"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    data = pd.read_parquet(args.dataset)
    reports: dict[int, dict[str, Any]] = {}
    for number, seed in enumerate(args.seeds, start=1):
        run_dir = args.output_dir / f"seed-{seed}"
        if not args.aggregate_only:
            print(f"[{number}/{len(args.seeds)}] B3 seed={seed}", file=sys.stderr, flush=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tfm_baseline3.cli",
                    "--dataset",
                    str(args.dataset),
                    "--output-dir",
                    str(run_dir),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
                stdout=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"B3 seed {seed} failed with {completed.returncode}")
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing seed result: {metrics_path}")
        reports[seed] = json.loads(metrics_path.read_text(encoding="utf-8"))

    payload = {
        "baseline": "B3",
        "model": "EdgeMLP",
        "dataset": {
            "path": str(args.dataset),
            "sha256": _sha256(args.dataset),
            "rows": int(len(data)),
        },
        "protocol": {"seeds": args.seeds, "paired_with_graphsage": True},
        "aggregate": {
            split: {
                metric: _summary([report[split][metric] for report in reports.values()])
                for metric in METRICS
            }
            for split in ("validation", "test")
        },
        "runs": {
            str(seed): {
                "best_epoch": report["training"]["best_epoch"],
                "validation_f1_incompatible": report["validation"]["f1_incompatible"],
                "test_f1_incompatible": report["test"]["f1_incompatible"],
            }
            for seed, report in reports.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) > 1 and not np.isclose(std, 0.0, atol=1e-15):
        interval = t.interval(
            0.95, len(array) - 1, loc=array.mean(), scale=std / np.sqrt(len(array))
        )
        confidence = [max(0.0, float(interval[0])), min(1.0, float(interval[1]))]
    else:
        confidence = [float(array.mean()), float(array.mean())]
    return {
        "mean": float(array.mean()),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "confidence_interval_95": confidence,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
