from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .layers import (
    BaselineArchitectureConfig,
    DoGPreprocessor,
    IntensityToLatencyEncoder,
    OutputClassGrouping,
    SpikingConvBlock,
)


class SpykeTorchBaselineNetwork(nn.Module):
    """
    Minimal paper-shaped baseline model.

    This module captures the shared topology described in the paper:
    DoG preprocessing -> latency coding -> S1/C1 -> S2/C2 -> S3/C3 -> grouped classes.

    The exact SpykeTorch spike-time dynamics and STDP rules belong in trainers
    and plasticity modules; this model keeps the structural contract stable so
    the rest of the project can start wiring against it.
    """

    def __init__(self, config: Optional[BaselineArchitectureConfig] = None) -> None:
        super().__init__()
        self.config = config or BaselineArchitectureConfig()
        self.config.validate()

        self.preprocessor = DoGPreprocessor(self.config)
        self.encoder = IntensityToLatencyEncoder()

        self.s1 = SpikingConvBlock(
            in_channels=self.config.dog_channels,
            out_channels=self.config.s1_maps,
            kernel_size=self.config.conv_kernel_size,
            pool_kernel_size=self.config.pool_kernel_sizes[0],
            pool_stride=self.config.pool_strides[0],
            inhibition_radius=self.config.inhibition_radii[0],
            weight_mean=self.config.weight_mean,
            weight_std=self.config.weight_std,
        )
        self.s2 = SpikingConvBlock(
            in_channels=self.config.s1_maps,
            out_channels=self.config.s2_maps,
            kernel_size=self.config.conv_kernel_size,
            pool_kernel_size=self.config.pool_kernel_sizes[1],
            pool_stride=self.config.pool_strides[1],
            inhibition_radius=self.config.inhibition_radii[1],
            weight_mean=self.config.weight_mean,
            weight_std=self.config.weight_std,
        )

        s3_in_features = self._infer_s3_in_features()
        self.s3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(s3_in_features, self.config.s3_neurons),
        )
        nn.init.normal_(
            self.s3[1].weight,
            mean=self.config.weight_mean,
            std=self.config.weight_std,
        )
        nn.init.zeros_(self.s3[1].bias)

        self.classifier = OutputClassGrouping(
            num_classes=self.config.num_classes,
            neurons_per_class=self.config.neurons_per_class,
        )

    def _infer_s3_in_features(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.config.input_channels,
                self.config.input_size,
                self.config.input_size,
            )
            dummy = self.preprocessor(dummy)
            dummy = self.encoder(dummy)
            dummy = self.s1(dummy)
            dummy = self.s2(dummy)
            return int(dummy.numel())

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        dog = self.preprocessor(x)
        latency = self.encoder(dog)
        s1 = self.s1(latency)
        s2 = self.s2(s1)
        s2_flat = torch.flatten(s2, start_dim=1)
        s3 = self.s3(s2)
        class_scores = self.classifier(s3)
        return {
            "dog": dog,
            "latency": latency,
            "s1": s1,
            "s2": s2,
            "s2_flat": s2_flat,
            "s3": s3,
            "class_scores": class_scores,
        }

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return features["class_scores"]

    def describe(self) -> Dict[str, Any]:
        return asdict(self.config)


def build_baseline_network(overrides: Optional[Dict[str, Any]] = None) -> SpykeTorchBaselineNetwork:
    config = BaselineArchitectureConfig(**(overrides or {}))
    return SpykeTorchBaselineNetwork(config=config)
