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
    weight_clip_min: float = 0.0
    weight_clip_max: float = 1.0
    active_threshold: float = 0.5
    reward_active: float = 0.004
    reward_inactive: float = -0.003
    punish_active: float = -0.003
    punish_inactive: float = 0.0005
    use_weight_stabilizer: bool = True


class RewardModulatedSTDP:
    """Reward-modulated output-layer STDP approximation.

    This keeps the class-group winner behavior from the paper-shaped S3/C3
    layer, then applies local reward/punishment updates to the winning output
    neuron. Active pre-synaptic features are updated differently from inactive
    features, and a bounded stabilizer prevents unbounded drift.
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
        grouped_logits = raw_logits.view(batch_size, num_classes, neurons_per_class)
        class_scores = grouped_logits.amax(dim=-1)
        predicted_classes = class_scores.argmax(dim=-1)

        reward_updates = 0
        punish_updates = 0
        correct = int((predicted_classes == targets).sum().item())
        missed = batch_size - correct
        phi_reward = missed / max(batch_size, 1)
        phi_punish = correct / max(batch_size, 1)

        for sample_idx in range(batch_size):
            feature_vec = features[sample_idx]
            target_class = int(targets[sample_idx].item())
            predicted_class = int(predicted_classes[sample_idx].item())
            active = self._active_mask(feature_vec)

            if predicted_class == target_class:
                target_start = target_class * neurons_per_class
                target_end = target_start + neurons_per_class
                target_group = raw_logits[sample_idx, target_start:target_end]
                winner_idx = target_start + int(target_group.argmax().item())
                self._apply_local_delta(
                    weights=weights,
                    neuron_idx=winner_idx,
                    active=active,
                    positive=float(self.config.reward_active) * float(self.config.reward_scale),
                    negative=float(self.config.reward_inactive) * float(self.config.reward_scale),
                    modulator=max(phi_reward, 1.0 / max(batch_size, 1)),
                )
                reward_updates += 1
            else:
                pred_start = predicted_class * neurons_per_class
                pred_end = pred_start + neurons_per_class
                pred_group = raw_logits[sample_idx, pred_start:pred_end]
                winner_idx = pred_start + int(pred_group.argmax().item())
                self._apply_local_delta(
                    weights=weights,
                    neuron_idx=winner_idx,
                    active=active,
                    positive=float(self.config.punish_active) * float(self.config.punish_scale),
                    negative=float(self.config.punish_inactive) * float(self.config.punish_scale),
                    modulator=max(phi_punish, 1.0 / max(batch_size, 1)),
                )
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
            "correct_before_update": correct,
            "phi_reward": float(phi_reward),
            "phi_punish": float(phi_punish),
        }

    def _active_mask(self, feature_vec: Tensor) -> Tensor:
        max_value = feature_vec.max().clamp_min(1e-8)
        threshold = max_value * float(self.config.active_threshold)
        return feature_vec >= threshold

    def _apply_local_delta(
        self,
        weights: Tensor,
        neuron_idx: int,
        active: Tensor,
        positive: float,
        negative: float,
        modulator: float,
    ) -> None:
        row = weights[neuron_idx]
        if self.config.use_weight_stabilizer:
            stabilizer = (row - self.config.weight_clip_min) * (self.config.weight_clip_max - row)
        else:
            stabilizer = torch.ones_like(row)
        delta = torch.where(
            active,
            torch.full_like(row, positive),
            torch.full_like(row, negative),
        )
        row.add_(float(self.config.learning_rate) * float(modulator) * delta * stabilizer)