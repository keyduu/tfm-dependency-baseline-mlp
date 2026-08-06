from tfm_baseline3.inference_benchmark import run_inference_benchmark
from tfm_baseline3.training import TrainingConfig, train_experiment


def test_inference_benchmark_reuses_saved_checkpoint(sample_dataset, tmp_path) -> None:
    run_dir = tmp_path / "trained/seed-42"
    train_experiment(
        sample_dataset,
        run_dir,
        config=TrainingConfig(
            hidden_channels=8,
            dropout=0,
            max_epochs=1,
            patience=1,
            temporal_batch_size=6,
            device="cpu",
        ),
    )
    dataset = tmp_path / "dataset.parquet"
    sample_dataset.to_parquet(dataset, index=False)
    output = tmp_path / "inference.json"

    report = run_inference_benchmark(
        sample_dataset,
        dataset,
        tmp_path / "trained",
        output,
        device_name="cpu",
        warmup_runs=1,
        measured_runs=2,
    )

    aggregate = report["aggregate"]["with_vb_in_range"]
    assert aggregate["saved_models"] == 1
    assert aggregate["measured_runs"] == 2
    assert report["runs"][0]["predictions_verified"] is True
    assert output.is_file()
