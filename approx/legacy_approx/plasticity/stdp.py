from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LocalSTDPConfig:
    a_plus: float = 0.004
    a_minus: float = -0.003
    lr_multiply_every: int = 500
    lr_multiply_factor: float = 2.0
    max_a_plus: float = 0.15
    min_a_minus: float = -0.1125
    weight_min: float = 0.0
    weight_max: float = 1.0
    active_threshold: float = 0.5
    winners_per_sample: int = 1
    max_updates_per_batch: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalConvSTDP:
    """Layer-wise convolutional STDP approximation for S1/S2 training.

    The original SpykeTorch implementation uses rank-order spike timing. This
    updater keeps the same local-learning shape in the current dense-tensor
    code: select winning post-synaptic responses, extract their pre-synaptic
    receptive fields, potentiate active pre-synaptic entries, depress inactive
    entries, and apply a bounded weight stabilizer.
    """

    def __init__(self, config: LocalSTDPConfig) -> None:
        self.config = config
        self.step_count = 0
        self.current_a_plus = float(config.a_plus)
        self.current_a_minus = float(config.a_minus)

    @torch.no_grad()
    def update_conv_weights(self, conv: nn.Conv2d, inputs: Tensor) -> Dict[str, Any]:
        if inputs.ndim != 4:
            raise ValueError("Expected NCHW inputs for convolutional STDP.")

        responses = F.conv2d(
            inputs,
            conv.weight.data,
            bias=conv.bias.data if conv.bias is not None else None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )
        batch_size, out_channels, out_h, out_w = responses.shape
        flat = responses.view(batch_size, -1)
        winners_per_sample = max(1, min(int(self.config.winners_per_sample), flat.shape[1]))
        _, winner_indices = torch.topk(flat, k=winners_per_sample, dim=1)

        updates = 0
        max_updates = max(1, int(self.config.max_updates_per_batch))
        kernel_h, kernel_w = conv.kernel_size
        stride_h, stride_w = conv.stride
        dilation_h, dilation_w = conv.dilation
        pad_h, pad_w = conv.padding
        padded = F.pad(inputs, (pad_w, pad_w, pad_h, pad_h))

        for sample_idx in range(batch_size):
            for winner_flat_idx in winner_indices[sample_idx].tolist():
                if updates >= max_updates:
                    break
                out_channel = winner_flat_idx // (out_h * out_w)
                spatial_idx = winner_flat_idx % (out_h * out_w)
                out_y = spatial_idx // out_w
                out_x = spatial_idx % out_w
                in_y = out_y * stride_h
                in_x = out_x * stride_w

                patch = padded[
                    sample_idx,
                    :,
                    in_y : in_y + dilation_h * (kernel_h - 1) + 1 : dilation_h,
                    in_x : in_x + dilation_w * (kernel_w - 1) + 1 : dilation_w,
                ]
                if patch.shape != conv.weight.data[out_channel].shape:
                    continue

                patch_max = patch.max().clamp_min(1e-8)
                patch_mean = patch.mean()
                active = patch >= torch.maximum(
                    patch_max * float(self.config.active_threshold),
                    patch_mean,
                )
                weight = conv.weight.data[out_channel]
                stabilizer = (weight - self.config.weight_min) * (self.config.weight_max - weight)
                delta = torch.where(
                    active,
                    torch.full_like(weight, self.current_a_plus),
                    torch.full_like(weight, self.current_a_minus),
                )
                weight.add_(delta * stabilizer)
                updates += 1
            if updates >= max_updates:
                break

        conv.weight.data.clamp_(self.config.weight_min, self.config.weight_max)
        self.step_count += 1
        self._maybe_adjust_rates()
        return {
            "updates": updates,
            "a_plus": self.current_a_plus,
            "a_minus": self.current_a_minus,
            "winner_responses_mean": float(responses.mean().item()),
            "winner_responses_max": float(responses.max().item()),
        }

    def _maybe_adjust_rates(self) -> None:
        every = int(self.config.lr_multiply_every)
        if every <= 0 or self.step_count % every != 0:
            return
        self.current_a_plus = min(
            self.current_a_plus * float(self.config.lr_multiply_factor),
            float(self.config.max_a_plus),
        )
        self.current_a_minus = max(
            self.current_a_minus * float(self.config.lr_multiply_factor),
            float(self.config.min_a_minus),
        )