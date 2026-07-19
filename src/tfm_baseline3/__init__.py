"""Neural edge-only baseline B3."""

from tfm_baseline3.model import EdgeMLPClassifier
from tfm_baseline3.training import TrainingConfig, train_experiment

__all__ = ["EdgeMLPClassifier", "TrainingConfig", "train_experiment"]
