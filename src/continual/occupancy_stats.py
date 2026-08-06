from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


def max_normalize(values: Tensor, eps: float = 1e-8) -> Tensor:
    values = values.clamp(min=0.0)
    peak = float(values.max().item()) if values.numel() else 0.0
    if peak <= eps:
        return torch.zeros_like(values)
    return values / peak


def as_2d_label_counts(
    winner_label_counts: Optional[Union[Sequence[Sequence[int]], Tensor]],
    num_neurons: int,
    num_classes: int,
) -> Tensor:
    if winner_label_counts is None:
        return torch.zeros(num_neurons, num_classes, dtype=torch.float32)

    if isinstance(winner_label_counts, Tensor):
        counts = winner_label_counts.detach().float().cpu()
    else:
        counts = torch.tensor(list(winner_label_counts), dtype=torch.float32)

    if counts.ndim == 1:
        if counts.numel() != num_neurons * num_classes:
            raise ValueError(
                "1D winner_label_counts must have length num_neurons * num_classes "
                f"({num_neurons * num_classes}), got {counts.numel()}."
            )
        counts = counts.reshape(num_neurons, num_classes)
    elif counts.ndim != 2:
        raise ValueError("winner_label_counts must be 2D [num_neurons, num_classes].")

    if counts.shape[0] < num_neurons:
        padded = torch.zeros(num_neurons, counts.shape[1], dtype=torch.float32)
        padded[: counts.shape[0]] = counts
        counts = padded
    elif counts.shape[0] > num_neurons:
        counts = counts[:num_neurons]

    if counts.shape[1] < num_classes:
        padded = torch.zeros(counts.shape[0], num_classes, dtype=torch.float32)
        padded[:, : counts.shape[1]] = counts
        counts = padded
    elif counts.shape[1] > num_classes:
        counts = counts[:, :num_classes]

    return counts


def winner_counts_tensor(winner_counts: Sequence[int], num_neurons: int) -> Tensor:
    return torch.tensor(
        [int(winner_counts[idx]) if idx < len(winner_counts) else 0 for idx in range(num_neurons)],
        dtype=torch.float32,
    )


def compute_selectivity(label_counts: Tensor) -> Tuple[Tensor, Tensor]:
    totals = label_counts.sum(dim=1)
    safe_totals = totals.clamp(min=1.0)
    dominant_counts, dominant_labels = label_counts.max(dim=1)
    q_i = dominant_counts / safe_totals
    q_i = torch.where(totals > 0, q_i, torch.zeros_like(q_i))
    return q_i, dominant_labels


def compute_neuron_weight_importance(conv3_weight: Tensor, use_weight_strength: bool) -> Tensor:
    if not use_weight_strength:
        return torch.ones(conv3_weight.shape[0], dtype=torch.float32)
    per_neuron = conv3_weight.detach().float().cpu().abs().reshape(conv3_weight.shape[0], -1).mean(dim=1)
    return max_normalize(per_neuron)


def infer_label_counts_from_decision_map(
    winner_counts: Tensor,
    decision_map: Sequence[int],
    num_classes: int,
) -> Tensor:
    num_neurons = int(winner_counts.numel())
    label_counts = torch.zeros(num_neurons, num_classes, dtype=torch.float32)
    for neuron_idx in range(num_neurons):
        wins = float(winner_counts[neuron_idx].item())
        if wins <= 0:
            continue
        if neuron_idx >= len(decision_map):
            continue
        mapped_label = int(decision_map[neuron_idx])
        if 0 <= mapped_label < num_classes:
            label_counts[neuron_idx, mapped_label] = wins
    return label_counts


@dataclass
class Task1OccupancyStats:
    """Shared Task 1 memory-occupancy statistics for partition and SDPM."""

    f_i: Tensor
    q_i: Tensor
    I_i: Tensor
    dominant_labels: Tensor

    def neuron_occupancy_score(
        self,
        *,
        use_frequency: bool = True,
        use_selectivity: bool = True,
        use_weight_strength: bool = True,
    ) -> Tensor:
        score = torch.ones_like(self.f_i)
        if use_frequency:
            score = score * max_normalize(self.f_i)
        if use_selectivity:
            score = score * self.q_i
        if use_weight_strength:
            score = score * self.I_i
        return score

    def combined_score(self) -> Tensor:
        """Default occupancy score: normalize(f_i) * q_i * I_i."""
        return self.neuron_occupancy_score(
            use_frequency=True,
            use_selectivity=True,
            use_weight_strength=True,
        )


def fit_task1_occupancy_stats(
    *,
    conv3_weight: Tensor,
    winner_counts: Sequence[int],
    num_classes: int,
    winner_label_counts: Optional[Union[Sequence[Sequence[int]], Tensor]] = None,
    decision_map: Optional[Sequence[int]] = None,
    use_weight_strength: bool = True,
    allow_decision_map_fallback: bool = True,
) -> Task1OccupancyStats:
    num_neurons = int(conv3_weight.shape[0])
    f_i = winner_counts_tensor(winner_counts, num_neurons)
    label_counts = as_2d_label_counts(winner_label_counts, num_neurons, num_classes)
    if float(label_counts.sum().item()) <= 0.0 and allow_decision_map_fallback and decision_map is not None:
        label_counts = infer_label_counts_from_decision_map(f_i, decision_map, num_classes)

    q_i, dominant_labels = compute_selectivity(label_counts)
    I_i = compute_neuron_weight_importance(conv3_weight, use_weight_strength)
    return Task1OccupancyStats(
        f_i=f_i,
        q_i=q_i,
        I_i=I_i,
        dominant_labels=dominant_labels,
    )
