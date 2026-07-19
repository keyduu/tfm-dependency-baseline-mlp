import pytest

from tfm_baseline3.schema import validate_dataset


def test_accepts_temporal_splits(sample_dataset) -> None:
    data = validate_dataset(sample_dataset)
    assert list(data["split"].drop_duplicates()) == ["train", "val", "test"]


def test_rejects_temporal_overlap(sample_dataset) -> None:
    sample_dataset.loc[sample_dataset["split"].eq("train"), "va_upload_time"] = (
        "2024-03-01T00:00:00+00:00"
    )
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_dataset(sample_dataset)
