from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    rows = []
    starts = {
        "train": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "val": datetime(2024, 2, 1, tzinfo=timezone.utc),
        "test": datetime(2024, 3, 1, tzinfo=timezone.utc),
    }
    for split, start in starts.items():
        for index in range(12):
            incompatible = index % 3 == 0
            rows.append(
                {
                    "pair_id": f"{split}-{index}",
                    "pa": f"parent-{index % 4}",
                    "va": f"1.{index % 3}.0",
                    "pb": f"dependency-{index % 3}",
                    "vb": "2.0.0" if incompatible else f"1.{index % 5}.0",
                    "va_upload_time": start + timedelta(hours=index),
                    "vb_upload_time": start - timedelta(days=2) + timedelta(hours=index),
                    "declared_range": ">=1,<2",
                    "dep_type": "direct",
                    "vb_in_range": not incompatible,
                    "va_num_prev_releases": index,
                    "va_size_bytes": 1000 + index,
                    "va_num_deps": 3,
                    "vb_num_prev_releases": index + 1,
                    "vb_size_bytes": 2000 + index,
                    "vb_num_deps": 1,
                    "label": int(incompatible),
                    "label_reason": "fixture",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)
