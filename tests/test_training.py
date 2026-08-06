import json

import pandas as pd

from tfm_baseline3.training import TrainingConfig, train_experiment


def test_training_writes_independent_b3_artifacts(sample_dataset, tmp_path) -> None:
    report = train_experiment(
        sample_dataset,
        tmp_path,
        config=TrainingConfig(
            hidden_channels=8,
            dropout=0,
            max_epochs=2,
            patience=1,
            temporal_batch_size=6,
            device="cpu",
        ),
    )

    assert report["baseline"] == "B3"
    assert report["features"]["uses_graph"] is False
    assert report["features"]["vb_in_range_included"] is True
    assert "vb_in_range" in report["features"]["numeric"]
    assert (tmp_path / "models/baseline3_mlp.pt").is_file()
    assert (tmp_path / "models/preprocessor.joblib").is_file()
    predictions = pd.read_parquet(tmp_path / "predictions/baseline3_test.parquet")
    assert len(predictions) == 12
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["seed"] == 42
    assert report["environment"]["cpu"]["logical_cores"] >= 1
    assert report["environment"]["cuda"]["available"] in {True, False}


def test_training_can_exclude_vb_in_range(sample_dataset, tmp_path) -> None:
    report = train_experiment(
        sample_dataset,
        tmp_path,
        config=TrainingConfig(
            hidden_channels=8,
            dropout=0,
            max_epochs=1,
            patience=1,
            temporal_batch_size=6,
            device="cpu",
            include_vb_in_range=False,
        ),
    )

    assert report["config"]["include_vb_in_range"] is False
    assert report["features"]["vb_in_range_included"] is False
    assert "vb_in_range" not in report["features"]["numeric"]
