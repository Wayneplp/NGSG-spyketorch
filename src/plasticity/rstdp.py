from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn

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


class SpykeTorchRewardSTDP(nn.Module):
    """Tutorial-style R-STDP: official SpykeTorch STDP plus anti-STDP."""

    def __init__(self, conv_layer: snn.Convolution, config: SpykeTorchRSTDPConfig) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.config = config
        self.stdp = snn.STDP(
            conv_layer=conv_layer,
            learning_rate=(
                config.reward_active * config.reward_scale,
                config.reward_inactive * config.reward_scale,
            ),
            use_stabilizer=config.use_stabilizer,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )
        self.anti_stdp = snn.STDP(
            conv_layer=conv_layer,
            learning_rate=(
                config.punish_active * config.punish_scale,
                config.punish_inactive * config.punish_scale,
            ),
            use_stabilizer=config.use_stabilizer,
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
        )

    @torch.no_grad()
    def update(
        self,
        input_spikes: Tensor,
        potentials: Tensor,
        output_spikes: Tensor,
        target: int,
        num_classes: int,
        neurons_per_class: int,
        kwta: int = 1,
        inhibition_radius: int = 0,
    ) -> Dict[str, Any]:
        winners = sf.get_k_winners(
            potentials,
            kwta=kwta,
            inhibition_radius=inhibition_radius,
            spikes=output_spikes,
        )
        if len(winners) == 0:
            return {
                "prediction": self._fallback_prediction(potentials, num_classes, neurons_per_class),
                "reward_updates": 0,
                "punish_updates": 0,
                "winner_count": 0,
            }

        target = int(target)
        predicted = int(winners[0][0] // neurons_per_class)
        correct = predicted == target

        if correct:
            self.stdp(input_spikes, potentials, output_spikes, winners=winners)
            return {
                "prediction": predicted,
                "reward_updates": 1,
                "punish_updates": 0,
                "winner_count": len(winners),
            }

        self.anti_stdp(input_spikes, potentials, output_spikes, winners=winners)
        return {
            "prediction": predicted,
            "reward_updates": 0,
            "punish_updates": 1,
            "winner_count": len(winners),
        }

    def update_learning_rate(
        self,
        reward_active: float,
        reward_inactive: float,
        punish_active: float,
        punish_inactive: float,
    ) -> None:
        self.stdp.update_all_learning_rate(reward_active, reward_inactive)
        self.anti_stdp.update_all_learning_rate(punish_active, punish_inactive)

    def _fallback_prediction(self, potentials: Tensor, num_classes: int, neurons_per_class: int) -> int:
        scores = potentials.view(potentials.shape[0], num_classes, neurons_per_class, -1).amax(dim=(0, 2, 3))
        return int(scores.argmax().item())

