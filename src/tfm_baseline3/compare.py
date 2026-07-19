"""Cross-project paired comparison between B3 and GraphSAGE summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import t, wilcoxon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare B3 with GraphSAGE")
    parser.add_argument(
        "--b3-summary",
        type=Path,
        default=Path("results/_local/baseline3/summary.json"),
    )
    parser.add_argument(
        "--graphsage-summary",
        type=Path,
        default=Path("results/_local/gnn/graphsage-seeds/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/_local/comparison/b3-vs-graphsage.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    b3 = json.loads(args.b3_summary.read_text(encoding="utf-8"))
    graphsage = json.loads(args.graphsage_summary.read_text(encoding="utf-8"))
    comparison = compare_summaries(b3, graphsage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def compare_summaries(b3: dict[str, Any], graphsage: dict[str, Any]) -> dict[str, Any]:
    if b3["dataset"]["sha256"] != graphsage["dataset"]["sha256"]:
        raise ValueError("B3 and GraphSAGE use different datasets")
    seeds = sorted(set(b3["runs"]).intersection(graphsage["runs"]), key=int)
    if seeds != [str(seed) for seed in b3["protocol"]["seeds"]]:
        raise ValueError("B3 and GraphSAGE do not contain the same seeds")
    deltas = np.asarray(
        [
            graphsage["runs"][seed]["test_f1_incompatible"]
            - b3["runs"][seed]["test_f1_incompatible"]
            for seed in seeds
        ]
    )
    mean = float(deltas.mean())
    std = float(deltas.std(ddof=1))
    interval = t.interval(0.95, len(deltas) - 1, loc=mean, scale=std / np.sqrt(len(deltas)))
    test = wilcoxon(deltas, alternative="two-sided", method="exact")
    return {
        "comparison": "GraphSAGE - B3",
        "dataset_sha256": b3["dataset"]["sha256"],
        "metric": "test_f1_incompatible",
        "seeds": [int(seed) for seed in seeds],
        "delta": {
            "mean": mean,
            "std": std,
            "confidence_interval_95": [float(interval[0]), float(interval[1])],
            "per_seed": {seed: float(delta) for seed, delta in zip(seeds, deltas)},
        },
        "wins": {
            "graphsage": int(np.sum(deltas > 0)),
            "ties": int(np.sum(deltas == 0)),
            "b3": int(np.sum(deltas < 0)),
        },
        "wilcoxon_two_sided_exact": {
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
