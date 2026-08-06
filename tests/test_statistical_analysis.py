import json

import numpy as np
import pandas as pd
import pytest

from tfm_baseline3.statistical_analysis import (
    calibration_table,
    holm_adjust,
    main,
    paired_bootstrap,
    prediction_metrics,
)


def test_prediction_metrics_include_ranking_and_calibration_scores() -> None:
    labels = np.asarray([1] * 20 + [0] * 40, dtype="int8")
    risk = np.linspace(0.99, 0.01, len(labels))
    predictions = _predictions(labels, risk)

    metrics, _ = prediction_metrics(predictions)

    assert metrics["precision_at_10"] == 1.0
    assert metrics["precision_at_50"] == pytest.approx(0.4)
    assert 0 <= metrics["brier"] <= 1
    assert metrics["log_loss"] >= 0
    assert 0 <= metrics["ece"] <= 1


def test_calibration_table_uses_defined_equal_width_bins() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype="int8")
    risk = np.asarray([0.1, 0.2, 0.8, 0.9])

    table = calibration_table(labels, risk, [0.0, 0.5, 1.0])

    assert table[0]["count"] == 2
    assert table[0]["mean_risk"] == pytest.approx(0.15)
    assert table[0]["positive_rate"] == 0.0
    assert table[1]["count"] == 2
    assert table[1]["mean_risk"] == pytest.approx(0.85)
    assert sum(row["ece_contribution"] for row in table) == pytest.approx(0.15)


def test_paired_bootstrap_is_deterministic() -> None:
    labels = np.asarray([1] * 10 + [0] * 10, dtype="int8")
    left = _predictions(labels, np.linspace(0.8, 0.2, len(labels)))
    right = _predictions(labels, np.linspace(0.95, 0.05, len(labels)))

    first = paired_bootstrap(
        left,
        right,
        metrics=("f1_incompatible", "brier"),
        bins=[0.0, 0.5, 1.0],
        resamples=50,
        seed=123,
    )
    second = paired_bootstrap(
        left,
        right,
        metrics=("f1_incompatible", "brier"),
        bins=[0.0, 0.5, 1.0],
        resamples=50,
        seed=123,
    )

    assert first == second
    assert first["metrics"]["brier"]["estimate"] < 0


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cli_writes_reproducible_tables(tmp_path) -> None:
    config = _write_analysis_fixture(tmp_path)
    output = tmp_path / "analysis"

    assert (
        main(
            [
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--bootstrap-resamples",
                "10",
                "--bootstrap-seed",
                "7",
                "--ece-bins",
                "5",
            ]
        )
        == 0
    )

    report = json.loads((output / "analysis.json").read_text())
    assert report["dataset_sha256"] == "fixture-sha"
    assert report["seeds"] == [1, 2]
    assert report["methods"]["paired_seed_test"]["family_size"] == 6
    assert report["comparisons"]["right-minus-left"]["delta_definition"] == "right_minus_left"
    assert (output / "per_seed_metrics.csv").is_file()
    assert (output / "calibration.csv").is_file()
    assert (output / "paired_bootstrap.csv").is_file()
    assert (output / "wilcoxon_holm.csv").is_file()


def test_cli_rejects_different_pair_ids(tmp_path) -> None:
    config = _write_analysis_fixture(tmp_path)
    path = tmp_path / "right" / "seed-2" / "predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[0, "pair_id"] = "different"
    predictions.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="different pair_ids"):
        main(
            [
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "analysis"),
                "--bootstrap-resamples",
                "2",
            ]
        )


def _write_analysis_fixture(tmp_path) -> object:
    labels = np.asarray([1] * 10 + [0] * 10, dtype="int8")
    for name, risk in (
        ("left", np.linspace(0.8, 0.2, len(labels))),
        ("right", np.linspace(0.95, 0.05, len(labels))),
    ):
        root = tmp_path / name
        (root / "summary.json").parent.mkdir(parents=True)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dataset": {"sha256": "fixture-sha"},
                    **(
                        {"protocol": {"seeds": [1, 2]}}
                        if name == "left"
                        else {"protocol": "fixture", "seeds": [1, 2]}
                    ),
                }
            )
        )
        for seed in (1, 2):
            path = root / f"seed-{seed}" / "predictions.parquet"
            path.parent.mkdir(parents=True)
            _predictions(labels, risk).to_parquet(path, index=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "name": "left",
                        "root": "left",
                        "prediction_pattern": "seed-{seed}/predictions.parquet",
                        "columns": {"risk": "risk"},
                    },
                    {
                        "name": "right",
                        "root": "right",
                        "prediction_pattern": "seed-{seed}/predictions.parquet",
                        "columns": {"risk": "risk"},
                    },
                ],
                "comparisons": [
                    {"name": "right-minus-left", "left": "left", "right": "right"}
                ],
            }
        )
    )
    return config


def _predictions(labels: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": [f"pair-{index:03d}" for index in range(len(labels))],
            "label": labels,
            "risk": risk,
            "prediction": (risk >= 0.5).astype("int8"),
        }
    )
