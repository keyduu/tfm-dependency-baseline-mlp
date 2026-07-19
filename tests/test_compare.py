import pytest

from tfm_baseline3.compare import compare_summaries


def test_compare_summaries_is_paired_by_seed() -> None:
    b3 = _summary([0.8, 0.9])
    graph = _summary([0.85, 0.88])

    result = compare_summaries(b3, graph)

    assert result["delta"]["mean"] == pytest.approx(0.015)
    assert result["wins"] == {"graphsage": 1, "ties": 0, "b3": 1}


def _summary(values: list[float]) -> dict:
    return {
        "dataset": {"sha256": "same"},
        "protocol": {"seeds": [1, 2]},
        "runs": {str(seed): {"test_f1_incompatible": value} for seed, value in zip((1, 2), values)},
    }
