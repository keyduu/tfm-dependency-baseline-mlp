"""Neural local-edge classifier used as baseline B3."""

from __future__ import annotations

import torch
from torch import nn


class EdgeMLPClassifier(nn.Module):
    def __init__(
        self,
        edge_features: int,
        *,
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.network(edge_features).squeeze(1)
