from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .neuron_partition import NeuronPartition, NeuronPartitionConfig
from .occupancy_stats import Task1OccupancyStats, as_2d_label_counts, compute_selectivity


def random_neuron_mask(
    num_neurons: int,
    k: int,
    seed: int,
    *,
    candidate_indices: Optional[Tensor] = None,
) -> Tensor:
    if k <= 0:
        return torch.zeros(num_neurons, dtype=torch.bool)

    if candidate_indices is None:
        candidates = torch.arange(num_neurons, dtype=torch.long)
    else:
        candidates = candidate_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
        candidates = candidates[(candidates >= 0) & (candidates < num_neurons)]
        candidates = torch.unique(candidates, sorted=True)

    if int(candidates.numel()) == 0:
        return torch.zeros(num_neurons, dtype=torch.bool)
    if k >= int(candidates.numel()) and int(candidates.numel()) == num_neurons:
        return torch.ones(num_neurons, dtype=torch.bool)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(int(candidates.numel()), generator=generator)
    selected = candidates[perm[: min(k, int(candidates.numel()))]]
    mask = torch.zeros(num_neurons, dtype=torch.bool)
    mask[selected] = True
    return mask


def shuffled_history_partition(
    partition: NeuronPartition,
    *,
    seed: int,
    winner_label_counts: Optional[Sequence[Sequence[int]]] = None,
    num_classes: int = 10,
) -> NeuronPartition:
    """Permute per-neuron winner histories, recompute q_i, and reassign roles."""
    num_neurons = partition.num_neurons
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_neurons, generator=generator)

    label_counts = as_2d_label_counts(
        winner_label_counts,
        num_neurons,
        num_classes,
    )
    if float(label_counts.sum().item()) <= 0.0:
        label_counts = torch.zeros(num_neurons, num_classes, dtype=torch.float32)
        for neuron_idx in range(num_neurons):
            wins = float(partition.f_i[neuron_idx].item())
            dom = int(partition.dominant_labels[neuron_idx].item())
            if wins > 0 and 0 <= dom < num_classes:
                label_counts[neuron_idx, dom] = wins

    shuffled_counts = label_counts[perm]
    shuffled_f_i = partition.f_i[perm].clone()
    q_i, dominant_labels = compute_selectivity(shuffled_counts)

    occupancy = Task1OccupancyStats(
        f_i=shuffled_f_i,
        q_i=q_i,
        I_i=partition.I_i.clone(),
        dominant_labels=dominant_labels,
    )
    return NeuronPartition.fit_from_occupancy(occupancy, partition.config)


def label_shuffled_fixed_frequency_partition(
    partition: NeuronPartition,
    *,
    seed: int,
    winner_label_counts: Optional[Sequence[Sequence[int]]] = None,
    num_classes: int = 10,
) -> NeuronPartition:
    """Keep each neuron's f_i fixed, borrow another neuron's label distribution, and refit roles."""
    num_neurons = partition.num_neurons
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_neurons, generator=generator)

    label_counts = as_2d_label_counts(
        winner_label_counts,
        num_neurons,
        num_classes,
    )
    if float(label_counts.sum().item()) <= 0.0:
        label_counts = torch.zeros(num_neurons, num_classes, dtype=torch.float32)
        for neuron_idx in range(num_neurons):
            wins = float(partition.f_i[neuron_idx].item())
            dom = int(partition.dominant_labels[neuron_idx].item())
            if wins > 0 and 0 <= dom < num_classes:
                label_counts[neuron_idx, dom] = wins

    donor_counts = label_counts[perm].float()
    donor_totals = donor_counts.sum(dim=1, keepdim=True)
    target_totals = partition.f_i.float().reshape(-1, 1)
    shuffled_counts = torch.zeros_like(donor_counts)
    nonzero = donor_totals.reshape(-1) > 0
    if bool(nonzero.any()):
        shuffled_counts[nonzero] = donor_counts[nonzero] / donor_totals[nonzero].clamp_min(1.0) * target_totals[nonzero]
    q_i, dominant_labels = compute_selectivity(shuffled_counts)

    occupancy = Task1OccupancyStats(
        f_i=partition.f_i.clone(),
        q_i=q_i,
        I_i=partition.I_i.clone(),
        dominant_labels=dominant_labels,
    )
    return NeuronPartition.fit_from_occupancy(occupancy, partition.config)

def frequency_only_partition(partition: NeuronPartition) -> NeuronPartition:
    """Stable/shared/reserve from f_i percentiles only (ignore q_stable_min for stable)."""
    config = partition.config
    freq_config = NeuronPartitionConfig(
        enabled=config.enabled,
        f_stable_percentile=config.f_stable_percentile,
        f_shared_percentile=config.f_shared_percentile,
        q_stable_min=0.0,
        use_weight_strength=config.use_weight_strength,
        treat_dead_as_reserve=config.treat_dead_as_reserve,
        num_classes=config.num_classes,
    )
    return NeuronPartition.fit_from_occupancy(partition.occupancy, freq_config)


def counterfactual_allow_masks(
    partition: NeuronPartition,
    *,
    seeds: Iterable[int],
) -> Dict[str, List[Tuple[int, Tensor]]]:
    """Build random same-size stable-only and random mask-stable allow masks."""
    stable_count = int(partition.mask_for_role("stable").sum().item())
    num_neurons = partition.num_neurons
    active_indices = (partition.f_i > 0).nonzero(as_tuple=False).reshape(-1)
    outputs: Dict[str, List[Tuple[int, Tensor]]] = {
        "random_stable_only": [],
        "random_mask_stable": [],
        "matched_active_random_stable_only": [],
        "matched_active_random_mask_stable": [],
    }
    for seed in seeds:
        pick = random_neuron_mask(num_neurons, stable_count, seed)
        outputs["random_stable_only"].append((int(seed), pick))
        outputs["random_mask_stable"].append((int(seed), pick))

        active_pick = random_neuron_mask(
            num_neurons,
            stable_count,
            seed,
            candidate_indices=active_indices,
        )
        outputs["matched_active_random_stable_only"].append((int(seed), active_pick))
        outputs["matched_active_random_mask_stable"].append((int(seed), active_pick))
    return outputs
