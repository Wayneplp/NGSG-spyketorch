from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DoGKernelSpec:
    kernel_size: int
    sigma_1: float
    sigma_2: float


@dataclass(frozen=True)
class BaselineArchitectureConfig:
    input_size: int = 28
    input_channels: int = 1
    conv_kernel_size: int = 5
    s1_maps: int = 30
    s2_maps: int = 250
    s3_neurons: int = 200
    num_classes: int = 10
    neurons_per_class: int = 20
    pool_kernel_sizes: Tuple[int, int, int] = (2, 2, 2)
    pool_strides: Tuple[int, int, int] = (2, 2, 2)
    inhibition_radii: Tuple[int, int, int] = (0, 0, 0)
    dog_kernels: Tuple[DoGKernelSpec, ...] = field(
        default_factory=lambda: (
            DoGKernelSpec(kernel_size=3, sigma_1=3 / 9, sigma_2=6 / 9),
            DoGKernelSpec(kernel_size=7, sigma_1=7 / 9, sigma_2=14 / 9),
            DoGKernelSpec(kernel_size=13, sigma_1=13 / 9, sigma_2=26 / 9),
        )
    )
    weight_mean: float = 0.8
    weight_std: float = 0.05

    @property
    def dog_channels(self) -> int:
        # Three scales with on/off responses.
        return len(self.dog_kernels) * 2

    def validate(self) -> None:
        if self.s3_neurons != self.num_classes * self.neurons_per_class:
            raise ValueError(
                "s3_neurons must equal num_classes * neurons_per_class "
                f"({self.num_classes} * {self.neurons_per_class})."
            )
        if len(self.pool_kernel_sizes) != 3 or len(self.pool_strides) != 3:
            raise ValueError("Expected exactly three pooling stages.")
        if len(self.inhibition_radii) != 3:
            raise ValueError("Expected exactly three inhibition radii.")


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    center = (kernel_size - 1) / 2.0
    coords = coords - center
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel


def build_dog_filter_bank(
    specs: Sequence[DoGKernelSpec],
    in_channels: int,
    device: torch.device,
) -> List[Tuple[Tensor, int]]:
    filters: List[Tuple[Tensor, int]] = []
    for spec in specs:
        gaussian_1 = _gaussian_kernel2d(spec.kernel_size, spec.sigma_1, device)
        gaussian_2 = _gaussian_kernel2d(spec.kernel_size, spec.sigma_2, device)
        dog = gaussian_1 - gaussian_2
        on_center = dog
        off_center = -dog
        stacked = torch.stack([on_center, off_center], dim=0).unsqueeze(1)
        stacked = stacked.repeat(1, in_channels, 1, 1)
        filters.append((stacked, spec.kernel_size // 2))
    return filters


class DoGPreprocessor(nn.Module):
    def __init__(self, config: BaselineArchitectureConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for kernels, padding in build_dog_filter_bank(
            self.config.dog_kernels,
            in_channels=x.shape[1],
            device=x.device,
        ):
            outputs.append(F.conv2d(x, kernels, padding=padding))
        return torch.cat(outputs, dim=1)


class IntensityToLatencyEncoder(nn.Module):
    """
    Minimal surrogate for rank-order encoding.

    SpykeTorch eventually needs discrete spike times. For now we keep a dense
    tensor whose larger values mean earlier spikes after normalization.
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.amin(dim=(-2, -1), keepdim=True)
        scale = x.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        x = x / scale
        return 1.0 - x


class InhibitionProxy(nn.Module):
    """
    Lightweight placeholder for lateral inhibition.

    The paper toggles inhibition per layer. For the first pass we keep the
    behavior simple and deterministic: local competition is approximated by
    suppressing values below the per-map maximum within each sample.
    """

    def __init__(self, radius: int) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, x: Tensor) -> Tensor:
        if self.radius <= 0:
            return x
        winners = x.amax(dim=(-2, -1), keepdim=True)
        keep_mask = x >= winners
        return torch.where(keep_mask, x, torch.zeros_like(x))


class SpikingConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int,
        pool_stride: int,
        inhibition_radius: int,
        weight_mean: float,
        weight_std: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.inhibition = InhibitionProxy(radius=inhibition_radius)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.reset_parameters(weight_mean=weight_mean, weight_std=weight_std)

    def reset_parameters(self, weight_mean: float, weight_std: float) -> None:
        nn.init.normal_(self.conv.weight, mean=weight_mean, std=weight_std)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.inhibition(x)
        x = self.pool(x)
        return x


class OutputClassGrouping(nn.Module):
    def __init__(self, num_classes: int, neurons_per_class: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.neurons_per_class = neurons_per_class

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_classes, self.neurons_per_class)
        # The paper uses winner-take-all style output behavior. Max pooling over
        # each class group is a simple proxy until exact spike-time decoding is added.
        return x.amax(dim=-1)

