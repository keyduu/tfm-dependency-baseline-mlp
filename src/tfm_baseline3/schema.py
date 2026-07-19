"""Dataset contract shared with the graph experiment."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "pair_id",
    "pa",
    "va",
    "pb",
    "vb",
    "va_upload_time",
    "vb_upload_time",
    "declared_range",
    "dep_type",
    "vb_in_range",
    "va_num_prev_releases",
    "va_size_bytes",
    "va_num_deps",
    "vb_num_prev_releases",
    "vb_size_bytes",
    "vb_num_deps",
    "label",
    "split",
}
SPLITS = ("train", "val", "test")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = frame.copy()
    data["split"] = data["split"].astype(str).str.lower().replace({"validation": "val"})
    unknown = sorted(set(data["split"]).difference(SPLITS))
    if unknown:
        raise ValueError(f"Unknown split values: {unknown}")
    if data["pair_id"].isna().any() or data["pair_id"].duplicated().any():
        raise ValueError("pair_id must be present and unique")
    if data["label"].isna().any() or not set(data["label"].astype(int)).issubset({0, 1}):
        raise ValueError("Labels must be binary values 0 or 1")
    data["label"] = data["label"].astype("int8")
    for split in SPLITS:
        if set(data.loc[data["split"].eq(split), "label"]) != {0, 1}:
            raise ValueError(f"Split {split!r} must contain labels 0 and 1")
    for column in ("va_upload_time", "vb_upload_time"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"{column} contains invalid timestamps")
    data["event_time"] = data[["va_upload_time", "vb_upload_time"]].max(axis=1)
    _validate_temporal_order(data)
    split_order = pd.Categorical(data["split"], categories=SPLITS, ordered=True)
    return (
        data.assign(_split_order=split_order)
        .sort_values(["_split_order", "event_time", "pair_id"], kind="stable")
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def _validate_temporal_order(data: pd.DataFrame) -> None:
    bounds = {
        split: (
            data.loc[data["split"].eq(split), "event_time"].min(),
            data.loc[data["split"].eq(split), "event_time"].max(),
        )
        for split in SPLITS
    }
    if not bounds["train"][1] < bounds["val"][0]:
        raise ValueError("Temporal leakage between train and validation")
    if not bounds["val"][1] < bounds["test"][0]:
        raise ValueError("Temporal leakage between validation and test")
