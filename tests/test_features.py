from tfm_baseline3.features import raw_edge_features


def test_features_include_range_but_exclude_targets(sample_dataset) -> None:
    features = raw_edge_features(sample_dataset)

    assert "vb_in_range" in features
    assert {"label", "label_reason", "pa", "pb"}.isdisjoint(features.columns)
