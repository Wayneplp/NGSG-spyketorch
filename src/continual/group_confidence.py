from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch
from torch import Tensor


def compute_group_class_votes(
    neuron_scores: Tensor,
    decision_map: Sequence[int],
    role_mask: Tensor,
    *,
    num_classes: int,
) -> Tensor:
    """Aggregate per-class votes within a neuron group."""
    votes = torch.zeros(int(num_classes), dtype=neuron_scores.dtype, device=neuron_scores.device)
    if neuron_scores.numel() == 0 or not bool(role_mask.any()):
        return votes

    scores = neuron_scores.reshape(-1)
    for neuron_idx in range(int(scores.shape[0])):
        if neuron_idx >= len(role_mask) or not bool(role_mask[neuron_idx].item()):
            continue
        label = int(decision_map[neuron_idx])
        if 0 <= label < num_classes:
            votes[label] += scores[neuron_idx]
    return votes


def compute_group_confidence(
    neuron_scores: Tensor,
    decision_map: Sequence[int],
    role_mask: Tensor,
    *,
    num_classes: int,
    eps: float = 1e-8,
) -> Tuple[float, Tensor]:
    """
    confidence = (top1 - top2) / (total + eps)

    Returns (confidence, class_votes).
    """
    votes = compute_group_class_votes(
        neuron_scores,
        decision_map,
        role_mask,
        num_classes=num_classes,
    )
    total = float(votes.sum().item())
    if total <= eps or not torch.isfinite(votes).any():
        return 0.0, votes

    sorted_votes, _ = torch.sort(votes, descending=True)
    top1 = float(sorted_votes[0].item())
    top2 = float(sorted_votes[1].item()) if sorted_votes.numel() > 1 else 0.0
    confidence = (top1 - top2) / (total + eps)
    return float(confidence), votes


def compute_stable_reserve_confidences(
    neuron_scores: Tensor,
    partition,
    decision_map: Sequence[int],
    *,
    num_classes: int,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Return conf_stable and conf_reserve for task inference."""
    conf_stable, _ = compute_group_confidence(
        neuron_scores,
        decision_map,
        partition.mask_for_role("stable"),
        num_classes=num_classes,
        eps=eps,
    )
    conf_reserve, _ = compute_group_confidence(
        neuron_scores,
        decision_map,
        partition.mask_for_role("reserve"),
        num_classes=num_classes,
        eps=eps,
    )
    return {"stable": conf_stable, "reserve": conf_reserve}
