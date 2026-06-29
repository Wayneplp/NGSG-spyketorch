from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor

from SpykeTorch import functional as sf
from SpykeTorch import snn


@dataclass
class SpykeTorchRSTDPConfig:
    reward_active: float = 0.004
    reward_inactive: float = -0.003
    punish_active: float = -0.003
    punish_inactive: float = 0.0005
    reward_scale: float = 1.0
    punish_scale: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    use_stabilizer: bool = True


class SpykeTorchRewardSTDP:
    """Reward-modulated output update operating on SpykeTorch snn.Convolution."""

    def __init__(self, conv_layer: snn.Convolution, config: SpykeTorchRSTDPConfig) -> None:
        self.conv_layer = conv_layer
        self.config = config

    @torch.no_grad()
    def update(
        self,
        input_spikes: Tensor,
        potentials: Tensor,
        output_spikes: Tensor,
        target: int,
        num_classes: int,
        neurons_per_class: int,
    ) -> Dict[str, Any]:
        scores = potentials.view(potentials.shape[0], num_classes, neurons_per_class, -1).amax(dim=(0, 2, 3))
        predicted = int(scores.argmax().item())
        target = int(target)
        correct = predicted == target

        if correct:
            group_start = target * neurons_per_class
            group_potentials = potentials[:, group_start : group_start + neurons_per_class]
            winner_offset = self._winner_offset(group_potentials)
            neuron_idx = group_start + winner_offset
            self._apply_delta(
                neuron_idx=neuron_idx,
                input_spikes=input_spikes,
                active_delta=self.config.reward_active * self.config.reward_scale,
                inactive_delta=self.config.reward_inactive * self.config.reward_scale,
            )
            return {"prediction": predicted, "reward_updates": 1, "punish_updates": 0}

        group_start = predicted * neurons_per_class
        group_potentials = potentials[:, group_start : group_start + neurons_per_class]
        winner_offset = self._winner_offset(group_potentials)
        neuron_idx = group_start + winner_offset
        self._apply_delta(
            neuron_idx=neuron_idx,
            input_spikes=input_spikes,
            active_delta=self.config.punish_active * self.config.punish_scale,
            inactive_delta=self.config.punish_inactive * self.config.punish_scale,
        )
        return {"prediction": predicted, "reward_updates": 0, "punish_updates": 1}

    def _winner_offset(self, group_potentials: Tensor) -> int:
        flat_idx = int(group_potentials.reshape(-1).argmax().item())
        per_neuron = group_potentials.shape[0] * group_potentials.shape[2] * group_potentials.shape[3]
        return flat_idx // per_neuron

    def _apply_delta(
        self,
        neuron_idx: int,
        input_spikes: Tensor,
        active_delta: float,
        inactive_delta: float,
    ) -> None:
        pre_spiked = input_spikes.sum(dim=0).sign().bool()
        row = self.conv_layer.weight[neuron_idx]
        delta = torch.where(
            pre_spiked,
            torch.full_like(row, float(active_delta)),
            torch.full_like(row, float(inactive_delta)),
        )
        if self.config.use_stabilizer:
            delta = delta * (row - self.config.lower_bound) * (self.config.upper_bound - row)
        row.add_(delta)
        row.clamp_(self.config.lower_bound, self.config.upper_bound)