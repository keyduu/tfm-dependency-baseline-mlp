from tfm_baseline3.features import raw_edge_features


def test_features_include_range_but_exclude_targets(sample_dataset) -> None:
    features = raw_edge_features(sample_dataset)

    assert "vb_in_range" in features
    assert {"label", "label_reason", "pa", "pb"}.isdisjoint(features.columns)


def test_vb_in_range_ablation_removes_exactly_one_feature(sample_dataset) -> None:
    included = raw_edge_features(sample_dataset)
    excluded = raw_edge_features(sample_dataset, include_vb_in_range=False)

    assert set(included.columns) - set(excluded.columns) == {"vb_in_range"}
    assert set(excluded.columns) - set(included.columns) == set()
