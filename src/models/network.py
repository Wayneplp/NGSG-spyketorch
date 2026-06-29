from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SpykeTorch import functional as sf
from SpykeTorch import snn, utils


@dataclass(frozen=True)
class DoGKernelSpec:
    kernel_size: int
    sigma_1: float
    sigma_2: float


@dataclass(frozen=True)
class SpykeTorchArchitectureConfig:
    input_size: int = 28
    input_channels: int = 1
    time_steps: int = 15
    conv_kernel_size: int = 5
    s1_maps: int = 30
    s2_maps: int = 250
    s3_neurons: int = 200
    num_classes: int = 10
    neurons_per_class: int = 20
    pool_kernel_sizes: Tuple[int, int] = (2, 2)
    pool_strides: Tuple[int, int] = (2, 2)
    inhibition_radii: Tuple[int, int] = (0, 0)
    dog_kernels: Tuple[DoGKernelSpec, ...] = field(
        default_factory=lambda: (
            DoGKernelSpec(kernel_size=3, sigma_1=3 / 9, sigma_2=6 / 9),
            DoGKernelSpec(kernel_size=7, sigma_1=7 / 9, sigma_2=14 / 9),
            DoGKernelSpec(kernel_size=13, sigma_1=13 / 9, sigma_2=26 / 9),
        )
    )
    s1_threshold: float = 15.0
    s2_threshold: float = 10.0
    s3_threshold: float = 5.0
    weight_mean: float = 0.8
    weight_std: float = 0.05

    @property
    def dog_channels(self) -> int:
        return len(self.dog_kernels) * 2

    def validate(self) -> None:
        if self.s3_neurons != self.num_classes * self.neurons_per_class:
            raise ValueError(
                "s3_neurons must equal num_classes * neurons_per_class "
                f"({self.num_classes} * {self.neurons_per_class})."
            )
        if len(self.pool_kernel_sizes) < 2 or len(self.pool_strides) < 2:
            raise ValueError("Expected at least two pooling stages for S1/C1 and S2/C2.")
        if len(self.inhibition_radii) < 2:
            raise ValueError("Expected at least two inhibition radii for S1 and S2.")


class SpykeTorchBaselineNetwork(nn.Module):
    """Paper-shaped SNN implemented with the original SpykeTorch package."""

    def __init__(self, config: Optional[SpykeTorchArchitectureConfig] = None) -> None:
        super().__init__()
        self.config = config or SpykeTorchArchitectureConfig()
        self.config.validate()

        self.register_buffer("dog_weight", self._build_dog_weight())
        self.encoder = utils.Intensity2Latency(self.config.time_steps, to_spike=True)

        self.s1 = snn.Convolution(
            self.config.dog_channels,
            self.config.s1_maps,
            self.config.conv_kernel_size,
            weight_mean=self.config.weight_mean,
            weight_std=self.config.weight_std,
        )
        self.c1 = snn.Pooling(self.config.pool_kernel_sizes[0], self.config.pool_strides[0])
        self.s2 = snn.Convolution(
            self.config.s1_maps,
            self.config.s2_maps,
            self.config.conv_kernel_size,
            weight_mean=self.config.weight_mean,
            weight_std=self.config.weight_std,
        )
        self.c2 = snn.Pooling(self.config.pool_kernel_sizes[1], self.config.pool_strides[1])
        self.s3 = snn.Convolution(
            self.config.s2_maps,
            self.config.s3_neurons,
            self._infer_s3_kernel_size(),
            weight_mean=self.config.weight_mean,
            weight_std=self.config.weight_std,
        )

    def _build_dog_weight(self) -> Tensor:
        kernels = []
        max_kernel = max(spec.kernel_size for spec in self.config.dog_kernels)
        for spec in self.config.dog_kernels:
            kernel = utils.DoGKernel(spec.kernel_size, spec.sigma_1, spec.sigma_2)()
            pad = (max_kernel - spec.kernel_size) // 2
            kernel = F.pad(kernel, (pad, pad, pad, pad))
            kernels.append(kernel)
            kernels.append(-kernel)
        return torch.stack(kernels, dim=0).unsqueeze(1).float()

    def _infer_s3_kernel_size(self) -> int:
        size = self.config.input_size
        max_dog = max(spec.kernel_size for spec in self.config.dog_kernels)
        # DoG uses same padding, so image size is unchanged.
        _ = max_dog
        size = size - self.config.conv_kernel_size + 1
        size = (size - self.config.pool_kernel_sizes[0]) // self.config.pool_strides[0] + 1
        size = size - self.config.conv_kernel_size + 1
        size = (size - self.config.pool_kernel_sizes[1]) // self.config.pool_strides[1] + 1
        if size <= 0:
            raise ValueError(f"Invalid inferred S3 kernel size: {size}")
        return int(size)

    def preprocess(self, image: Tensor) -> Tensor:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError("Expected single image tensor with shape CxHxW.")
        image_batch = image.unsqueeze(0)
        padding = self.dog_weight.shape[-1] // 2
        filtered = F.conv2d(image_batch, self.dog_weight, padding=padding).squeeze(0)
        filtered = filtered.clamp_min(0.0)
        if filtered.max() > 0:
            filtered = filtered / filtered.max()
        return filtered.unsqueeze(0)

    def encode(self, image: Tensor) -> Tensor:
        intensities = self.preprocess(image)
        return self.encoder(intensities).to(image.device)

    def _fire(self, potentials: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
        spikes, thresholded = sf.fire(
            potentials,
            threshold=threshold,
            return_thresholded_potentials=True,
        )
        inhibited = sf.pointwise_inhibition(thresholded)
        return inhibited.sign(), inhibited

    def s1_step(self, input_spikes: Tensor) -> Dict[str, Tensor]:
        potentials = self.s1(input_spikes)
        spikes, thresholded = self._fire(potentials, self.config.s1_threshold)
        pooled = self.c1(spikes)
        return {"input": input_spikes, "potentials": thresholded, "spikes": spikes, "pooled": pooled}

    def s2_step(self, s1_pooled: Tensor) -> Dict[str, Tensor]:
        potentials = self.s2(s1_pooled)
        spikes, thresholded = self._fire(potentials, self.config.s2_threshold)
        pooled = self.c2(spikes)
        return {"input": s1_pooled, "potentials": thresholded, "spikes": spikes, "pooled": pooled}

    def s3_step(self, s2_pooled: Tensor) -> Dict[str, Tensor]:
        potentials = self.s3(s2_pooled)
        spikes, thresholded = self._fire(potentials, self.config.s3_threshold)
        return {"input": s2_pooled, "potentials": thresholded, "spikes": spikes}

    def forward_spikes(self, image: Tensor) -> Dict[str, Tensor]:
        encoded = self.encode(image)
        s1 = self.s1_step(encoded)
        s2 = self.s2_step(s1["pooled"])
        s3 = self.s3_step(s2["pooled"])
        return {"encoded": encoded, "s1": s1, "s2": s2, "s3": s3}

    def class_scores_from_s3(self, s3_potentials: Tensor) -> Tensor:
        grouped = s3_potentials.view(
            self.config.time_steps,
            self.config.num_classes,
            self.config.neurons_per_class,
            -1,
        )
        return grouped.amax(dim=(0, 2, 3))

    def predict_single(self, image: Tensor) -> int:
        features = self.forward_spikes(image)
        scores = self.class_scores_from_s3(features["s3"]["potentials"])
        return int(scores.argmax().item())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            return self.class_scores_from_s3(self.forward_spikes(x)["s3"]["potentials"]).unsqueeze(0)
        scores = [self.forward(sample.to(next(self.parameters()).device)).squeeze(0) for sample in x]
        return torch.stack(scores, dim=0)

    def describe(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        payload["implementation"] = "official_spyketorch_package"
        payload["spyketorch_modules"] = ["snn.Convolution", "snn.Pooling", "snn.STDP", "functional.fire"]
        return payload


def build_baseline_network(overrides: Optional[Dict[str, Any]] = None) -> SpykeTorchBaselineNetwork:
    config = SpykeTorchArchitectureConfig(**(overrides or {}))
    return SpykeTorchBaselineNetwork(config=config)
