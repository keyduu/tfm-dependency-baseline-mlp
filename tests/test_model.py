import torch

from tfm_baseline3.model import EdgeMLPClassifier


def test_mlp_scores_one_logit_per_edge() -> None:
    model = EdgeMLPClassifier(12, hidden_channels=8, dropout=0)

    assert model(torch.randn(5, 12)).shape == (5,)
