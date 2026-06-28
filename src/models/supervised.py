from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SupervisedMLPConfig:
    input_size: int = 28
    input_channels: int = 1
    num_classes: int = 10
    hidden_units: int = 256
    dropout: float = 0.0


class SupervisedMNISTMLP(nn.Module):
    """Small supervised network for validating the continual-learning protocol."""

    def __init__(self, config: SupervisedMLPConfig) -> None:
        super().__init__()
        self.config = config
        input_features = config.input_size * config.input_size * config.input_channels
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, config.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_units, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def describe(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        payload["architecture"] = "supervised_mlp"
        return payload


def build_supervised_mnist_network(overrides: Dict[str, Any] | None = None) -> SupervisedMNISTMLP:
    config = SupervisedMLPConfig(**(overrides or {}))
    return SupervisedMNISTMLP(config=config)
