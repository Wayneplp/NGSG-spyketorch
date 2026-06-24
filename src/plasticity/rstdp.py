from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor


@dataclass
class RSTDPConfig:
    learning_rate: float = 0.01
    reward_scale: float = 1.0
    punish_scale: float = 0.5
    weight_decay: float = 0.0
    freeze_fraction: float = 0.0
    langevin_noise_std: float = 0.0
    weight_clip_min: float = -1.0
    weight_clip_max: float = 1.5


class RewardModulatedSTDP:
    """
    Minimal runnable approximation of class-conditional local updates.

    We update the winning neuron within the target class group using the current
    feature vector, and optionally punish the winning neuron of an incorrect
    predicted class. This keeps the update local to layer activations/weights
    while remaining simple enough to run without the full SpykeTorch stack.
    """

    def __init__(self, config: RSTDPConfig) -> None:
        self.config = config
        self.frozen_mask: Optional[Tensor] = None

    def set_frozen_mask_from_weights(self, weights: Tensor) -> None:
        fraction = float(self.config.freeze_fraction)
        if fraction <= 0.0:
            self.frozen_mask = None
            return

        flat_abs = weights.detach().abs().flatten()
        k = int(flat_abs.numel() * fraction)
        if k <= 0:
            self.frozen_mask = None
            return

        threshold = torch.topk(flat_abs, k=k, largest=True).values.min()
        self.frozen_mask = weights.detach().abs() >= threshold

    @torch.no_grad()
    def update_output_weights(
        self,
        weights: Tensor,
        features: Tensor,
        raw_logits: Tensor,
        targets: Tensor,
        num_classes: int,
        neurons_per_class: int,
    ) -> Dict[str, Any]:
        batch_size = features.shape[0]
        original_weights = weights.detach().clone()
        predicted_classes = raw_logits.view(batch_size, num_classes, neurons_per_class).amax(dim=-1).argmax(dim=-1)

        reward_updates = 0
        punish_updates = 0

        for sample_idx in range(batch_size):
            feature_vec = features[sample_idx]
            target_class = int(targets[sample_idx].item())
            predicted_class = int(predicted_classes[sample_idx].item())

            target_start = target_class * neurons_per_class
            target_end = target_start + neurons_per_class
            target_group = raw_logits[sample_idx, target_start:target_end]
            target_winner_offset = int(target_group.argmax().item())
            target_winner_idx = target_start + target_winner_offset

            reward_delta = self.config.learning_rate * self.config.reward_scale * feature_vec
            weights[target_winner_idx] += reward_delta
            reward_updates += 1

            if predicted_class != target_class:
                pred_start = predicted_class * neurons_per_class
                pred_end = pred_start + neurons_per_class
                pred_group = raw_logits[sample_idx, pred_start:pred_end]
                pred_winner_offset = int(pred_group.argmax().item())
                pred_winner_idx = pred_start + pred_winner_offset
                punish_delta = self.config.learning_rate * self.config.punish_scale * feature_vec
                weights[pred_winner_idx] -= punish_delta
                punish_updates += 1

        if self.config.weight_decay > 0.0:
            weights.mul_(1.0 - self.config.weight_decay)

        if self.config.langevin_noise_std > 0.0:
            weights.add_(torch.randn_like(weights) * self.config.langevin_noise_std)

        if self.frozen_mask is not None:
            weights.copy_(torch.where(self.frozen_mask, original_weights, weights))

        weights.clamp_(self.config.weight_clip_min, self.config.weight_clip_max)

        return {
            "reward_updates": reward_updates,
            "punish_updates": punish_updates,
        }
