"""CLI for one B3 execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from tfm_baseline3.training import TrainingConfig, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train neural baseline B3")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/_local/processed/dataset_mvd_v2.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/_local/baseline3/single"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temporal-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    data = pd.read_parquet(args.dataset)
    report = train_experiment(
        data,
        args.output_dir,
        config=TrainingConfig(
            seed=args.seed,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            temporal_batch_size=args.temporal_batch_size,
            device=args.device,
        ),
        dataset_metadata={
            "path": str(args.dataset),
            "sha256": _sha256(args.dataset),
            "rows": int(len(data)),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
