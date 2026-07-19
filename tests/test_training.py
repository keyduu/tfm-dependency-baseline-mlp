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
    assert (tmp_path / "models/baseline3_mlp.pt").is_file()
    assert (tmp_path / "models/preprocessor.joblib").is_file()
    predictions = pd.read_parquet(tmp_path / "predictions/baseline3_test.parquet")
    assert len(predictions) == 12
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["seed"] == 42
