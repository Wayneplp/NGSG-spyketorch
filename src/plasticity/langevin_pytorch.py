"""Pure-PyTorch Langevin/STDP dynamics used by the Langevin protocol.

This module deliberately has no SpykeTorch dependency.  The local update is
implemented with tensor operations and accepts the same time-major tensors as
the historical training loop: ``[T, C, H, W]``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class PyTorchBrownianSTDP(nn.Module):
    """Local STDP update followed by optional Brownian diffusion.

    ``winners`` may contain ``(feature, y, x)`` tuples.  Only winning output
    features are changed, matching the sparse convolutional update used by the
    Langevin notebook.  Research snapshots are taken after STDP and before
    noise; quantile bounds are applied only to synapses changed by the update.
    """

    def __init__(self, conv_layer: nn.Module, learning_rate, *, lower=0.2, upper=0.8):
        super().__init__()
        self.conv_layer = conv_layer
        self.ap, self.an = float(learning_rate[0]), float(learning_rate[1])
        self.lower_bound, self.upper_bound = float(lower), float(upper)
        self.epsilon = 0.0
        self.brownian = False
        self.collect_samples = False
        self.samples: list[Tensor] = []
        self.sample_every = 100
        self._sample_counter = 0
        self._sample_counter_box: Optional[list] = None
        self.quantile_lower: Optional[Tensor] = None
        self.quantile_upper: Optional[Tensor] = None

    def update_all_learning_rate(self, ap, an):
        self.ap, self.an = float(ap), float(an)

    @property
    def learning_rate(self):
        """SpykeTorch-compatible view of the two scalar learning rates.

        The shared trainer reads ``learning_rate[0][0/1]`` for adaptive
        schedules and diagnostics.  The pure-PyTorch implementation stores
        the values as scalars, so expose a device-local tensor view without
        changing the update rule.
        """
        device = self.conv_layer.weight.device
        return [[
            torch.as_tensor(self.ap, dtype=torch.float32, device=device),
            torch.as_tensor(self.an, dtype=torch.float32, device=device),
        ]]

    def set_brownian(self, enabled: bool, epsilon: float):
        self.brownian, self.epsilon = bool(enabled), float(epsilon)

    def start_collection(self, sample_every=100, *, sample_counter_box=None):
        self.samples = []
        self.collect_samples = True
        self.sample_every = max(1, int(sample_every))
        self._sample_counter = 0
        self._sample_counter_box = sample_counter_box

    def finish_collection(self):
        self.collect_samples = False
        self._sample_counter_box = None
        if not self.samples:
            return None
        stack = torch.stack(self.samples, dim=0).float()
        lo, hi = torch.quantile(stack, 0.1, dim=0), torch.quantile(stack, 0.9, dim=0)
        n = len(self.samples)
        self.samples = []
        self.quantile_lower, self.quantile_upper = lo.detach(), hi.detach()
        return self.quantile_lower, self.quantile_upper, n

    def _winner_mask(self, winners, shape, device):
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        for winner in winners or ():
            feature = int(winner[0])
            if 0 <= feature < shape[0]:
                mask[feature] = True
        return mask

    def forward(self, input_spikes, potentials, output_spikes, winners=None, **kwargs):
        del kwargs
        before = self.conv_layer.weight.detach().clone()
        x = input_spikes.float()
        y = output_spikes.float()
        # Aggregate pre/post activity over time and spatial locations.
        pre = x.mean(dim=0) if x.ndim >= 4 else x
        post = y.mean(dim=0) if y.ndim >= 4 else y
        with torch.no_grad():
            delta = self.ap * post.mean(dim=(-2, -1), keepdim=True).unsqueeze(1) * pre.mean(dim=tuple(range(1, pre.ndim)), keepdim=True)
            delta = delta.expand_as(self.conv_layer.weight) if delta.numel() == self.conv_layer.weight.numel() else torch.zeros_like(self.conv_layer.weight)
            mask = self._winner_mask(winners, self.conv_layer.weight.shape[:1], self.conv_layer.weight.device)
            self.conv_layer.weight.add_(delta * mask.view(-1, *([1] * (self.conv_layer.weight.ndim - 1))))
            if self.collect_samples:
                counter = self._sample_counter_box[0] if self._sample_counter_box is not None else self._sample_counter
                if self._sample_counter_box is not None: self._sample_counter_box[0] = counter + 1
                else: self._sample_counter += 1
                if counter % self.sample_every == 0: self.samples.append(self.conv_layer.weight.detach().cpu().clone())
            if self.brownian and self.epsilon > 0: self.conv_layer.weight.add_(torch.randn_like(self.conv_layer.weight) * self.epsilon)
            changed = self.conv_layer.weight != before
            if self.quantile_lower is not None and self.quantile_upper is not None:
                lo, hi = self.quantile_lower.to(self.conv_layer.weight), self.quantile_upper.to(self.conv_layer.weight)
                clipped = torch.maximum(torch.minimum(self.conv_layer.weight, hi), lo)
                self.conv_layer.weight.copy_(torch.where(changed, clipped, self.conv_layer.weight))
            self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)
