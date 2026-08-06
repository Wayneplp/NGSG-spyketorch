from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .occupancy_stats import as_2d_label_counts, compute_selectivity
from .reserve_activation import potential_planes


class ReadoutStrategy(str, Enum):
    WTA_STATIC = "wta_static"
    WTA_DOMINANT = "wta_dominant"
    CLASS_MAX_STATIC = "class_max_static"
    CLASS_MAX_DOMINANT_T1 = "class_max_dominant_t1"
    CLASS_MAX_DOMINANT_T2 = "class_max_dominant_t2"
    CLASS_MAX_TASK_AWARE = "class_max_task_aware"


@dataclass(frozen=True)
class NeuronLabelMaps:
    """Per-neuron class assignments derived from training winner statistics."""

    static_map: Tuple[int, ...]
    dominant_task1: Tuple[int, ...]
    dominant_task2: Tuple[int, ...]
    task_aware_mnist: Tuple[int, ...]
    task_aware_emnist: Tuple[int, ...]
    num_classes: int

    @classmethod
    def from_training_stats(
        cls,
        *,
        decision_map: Sequence[int],
        winner_label_counts_task1: Optional[Union[Sequence[Sequence[int]], Tensor]],
        winner_label_counts_task2: Optional[Union[Sequence[Sequence[int]], Tensor]] = None,
        num_classes: int = 10,
    ) -> "NeuronLabelMaps":
        num_neurons = len(decision_map)
        static = tuple(int(x) for x in decision_map)

        def dominant_from(counts: Optional[Union[Sequence[Sequence[int]], Tensor]]) -> Tuple[int, ...]:
            label_counts = as_2d_label_counts(counts, num_neurons, num_classes)
            _, dominant = compute_selectivity(label_counts)
            mapped = []
            for neuron_idx in range(num_neurons):
                label = int(dominant[neuron_idx].item())
                if float(label_counts[neuron_idx].sum().item()) <= 0.0:
                    label = static[neuron_idx] if neuron_idx < len(static) else 0
                mapped.append(label)
            return tuple(mapped)

        dom_t1 = dominant_from(winner_label_counts_task1)
        dom_t2 = dominant_from(winner_label_counts_task2)

        task_mnist = []
        task_emnist = []
        counts_t1 = as_2d_label_counts(winner_label_counts_task1, num_neurons, num_classes)
        counts_t2 = as_2d_label_counts(winner_label_counts_task2, num_neurons, num_classes)
        for neuron_idx in range(num_neurons):
            if float(counts_t1[neuron_idx].sum().item()) > 0.0:
                task_mnist.append(dom_t1[neuron_idx])
            else:
                task_mnist.append(static[neuron_idx] if neuron_idx < len(static) else 0)

            if float(counts_t2[neuron_idx].sum().item()) > 0.0:
                task_emnist.append(dom_t2[neuron_idx])
            elif float(counts_t1[neuron_idx].sum().item()) > 0.0:
                task_emnist.append(dom_t1[neuron_idx])
            else:
                task_emnist.append(static[neuron_idx] if neuron_idx < len(static) else 0)

        return cls(
            static_map=static,
            dominant_task1=dom_t1,
            dominant_task2=dom_t2,
            task_aware_mnist=tuple(task_mnist),
            task_aware_emnist=tuple(task_emnist),
            num_classes=num_classes,
        )

    def mismatch_counts(self) -> dict[str, int]:
        active = sum(1 for idx, label in enumerate(self.dominant_task1) if label != self.static_map[idx])
        return {
            "dominant_t1_vs_static": active,
            "dominant_t2_vs_static": sum(
                1 for idx, label in enumerate(self.dominant_task2) if label != self.static_map[idx]
            ),
        }


def aggregate_neuron_scores(pot: Tensor, num_neurons: int) -> Tensor:
    """Return per-neuron scalar scores [num_neurons] from S3 potentials."""
    planes = potential_planes(pot.detach().float().cpu(), num_neurons)
    return planes.reshape(num_neurons, -1).max(dim=1).values


def predict_wta_decision_map(
    neuron_scores: Tensor,
    decision_map: Sequence[int],
    *,
    allow_mask: Optional[Tensor] = None,
) -> int:
    """Global WTA readout with optional per-neuron allow mask (group-only / group-masked)."""
    if neuron_scores.numel() == 0:
        return -1
    scores = neuron_scores.clone()
    if allow_mask is not None:
        if allow_mask.shape[0] != scores.shape[0]:
            raise ValueError("allow_mask length must match number of neurons.")
        scores = scores.masked_fill(~allow_mask, float("-inf"))
    if not torch.isfinite(scores).any():
        return -1
    winner = int(scores.argmax().item())
    if winner < 0 or winner >= len(decision_map):
        return -1
    return int(decision_map[winner])


def role_allow_mask(
    partition,
    *,
    mode: str,
    role: str,
) -> Tensor:
    """Build neuron allow mask for group-only or group-masked inference."""
    role_mask = partition.mask_for_role(role)
    if mode == "only":
        return role_mask
    if mode == "mask":
        return ~role_mask
    raise ValueError(f"Unknown role mask mode '{mode}'. Expected 'only' or 'mask'.")


PARTITION_GROUP_DIAGNOSES = (
    ("all_200", None, None),
    ("stable_only", "only", "stable"),
    ("shared_only", "only", "shared"),
    ("reserve_only", "only", "reserve"),
    ("mask_stable", "mask", "stable"),
    ("mask_shared", "mask", "shared"),
    ("mask_reserve", "mask", "reserve"),
)

PARTITION_GROUP_LABELS = {
    "all_200": "all 200 neurons (baseline)",
    "stable_only": "6.1 stable-only",
    "shared_only": "6.1 shared-only",
    "reserve_only": "6.1 reserve-only",
    "mask_stable": "6.2 mask-stable",
    "mask_shared": "6.2 mask-shared",
    "mask_reserve": "6.2 mask-reserve",
}


def predict_from_scores(
    neuron_scores: Tensor,
    *,
    strategy: ReadoutStrategy,
    label_maps: NeuronLabelMaps,
    neurons_per_class: int,
    eval_task: str,
) -> int:
    if neuron_scores.numel() == 0:
        return -1

    if strategy == ReadoutStrategy.WTA_STATIC:
        winner = int(neuron_scores.argmax().item())
        return label_maps.static_map[winner]

    if strategy == ReadoutStrategy.WTA_DOMINANT:
        winner = int(neuron_scores.argmax().item())
        return label_maps.dominant_task1[winner]

    if strategy == ReadoutStrategy.CLASS_MAX_STATIC:
        num_classes = label_maps.num_classes
        grouped = neuron_scores.reshape(num_classes, neurons_per_class)
        return int(grouped.max(dim=1).values.argmax().item())

    label_map = _resolve_class_label_map(strategy, label_maps, eval_task)
    class_scores = torch.full((label_maps.num_classes,), float("-inf"))
    for neuron_idx, score in enumerate(neuron_scores):
        label = label_map[neuron_idx]
        if 0 <= label < label_maps.num_classes:
            class_scores[label] = torch.maximum(class_scores[label], score)
    if not torch.isfinite(class_scores).any():
        return -1
    return int(class_scores.argmax().item())


def _resolve_class_label_map(
    strategy: ReadoutStrategy,
    label_maps: NeuronLabelMaps,
    eval_task: str,
) -> Tuple[int, ...]:
    if strategy == ReadoutStrategy.CLASS_MAX_DOMINANT_T1:
        return label_maps.dominant_task1
    if strategy == ReadoutStrategy.CLASS_MAX_DOMINANT_T2:
        return label_maps.dominant_task2
    if strategy == ReadoutStrategy.CLASS_MAX_TASK_AWARE:
        if eval_task in ("task1", "mnist", "initial_mnist_digits"):
            return label_maps.task_aware_mnist
        return label_maps.task_aware_emnist
    raise ValueError(f"Strategy {strategy} is not a class-max strategy.")


def predict_from_potentials(
    pot: Tensor,
    *,
    strategy: ReadoutStrategy,
    label_maps: NeuronLabelMaps,
    neurons_per_class: int,
    eval_task: str,
) -> int:
    num_neurons = len(label_maps.static_map)
    scores = aggregate_neuron_scores(pot, num_neurons)
    return predict_from_scores(
        scores,
        strategy=strategy,
        label_maps=label_maps,
        neurons_per_class=neurons_per_class,
        eval_task=eval_task,
    )


def extract_winner_label_counts(result: Mapping[str, Any], stage: str) -> Optional[Any]:
    trainer = (result.get("extra") or {}).get("trainer") or {}
    if stage == "task1":
        output = (trainer.get("task1_training") or {}).get("output_training") or {}
    else:
        output = (trainer.get("task2_training") or {}).get("output_training") or {}
    counts = output.get("winner_label_counts")
    return counts if counts else None


STRATEGY_LABELS = {
    ReadoutStrategy.WTA_STATIC: "global WTA + decision_map (current)",
    ReadoutStrategy.WTA_DOMINANT: "global WTA + Task1 dominant_label",
    ReadoutStrategy.CLASS_MAX_STATIC: "class-max + static blocks",
    ReadoutStrategy.CLASS_MAX_DOMINANT_T1: "class-max + Task1 dominant_label",
    ReadoutStrategy.CLASS_MAX_DOMINANT_T2: "class-max + Task2 dominant_label",
    ReadoutStrategy.CLASS_MAX_TASK_AWARE: "class-max + task-aware labels",
}

ALL_STRATEGIES = tuple(ReadoutStrategy)
