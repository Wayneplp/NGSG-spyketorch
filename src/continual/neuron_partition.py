from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


class NeuronRole(IntEnum):
    RESERVE = 0
    SHARED = 1
    STABLE = 2
    DEAD = 3


ROLE_NAMES: Dict[NeuronRole, str] = {
    NeuronRole.RESERVE: "reserve",
    NeuronRole.SHARED: "shared",
    NeuronRole.STABLE: "stable",
    NeuronRole.DEAD: "dead",
}


def _max_normalize(values: Tensor, eps: float = 1e-8) -> Tensor:
    values = values.clamp(min=0.0)
    peak = float(values.max().item()) if values.numel() else 0.0
    if peak <= eps:
        return torch.zeros_like(values)
    return values / peak


def _as_2d_label_counts(
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


def _percentile_threshold(values: Tensor, percentile: float) -> float:
    active = values[values > 0]
    if active.numel() == 0:
        return 0.0
    percentile = float(min(max(percentile, 0.0), 1.0))
    return float(torch.quantile(active, percentile).item())


def _winner_counts_tensor(winner_counts: Sequence[int], num_neurons: int) -> Tensor:
    counts = torch.tensor(
        [int(winner_counts[idx]) if idx < len(winner_counts) else 0 for idx in range(num_neurons)],
        dtype=torch.float32,
    )
    return counts


def _compute_selectivity(label_counts: Tensor) -> Tuple[Tensor, Tensor]:
    totals = label_counts.sum(dim=1)
    safe_totals = totals.clamp(min=1.0)
    dominant_counts, dominant_labels = label_counts.max(dim=1)
    q_i = dominant_counts / safe_totals
    q_i = torch.where(totals > 0, q_i, torch.zeros_like(q_i))
    return q_i, dominant_labels


def _compute_neuron_importance(conv3_weight: Tensor, use_weight_strength: bool) -> Tensor:
    if not use_weight_strength:
        return torch.ones(conv3_weight.shape[0], dtype=torch.float32)
    per_neuron = conv3_weight.detach().float().cpu().abs().reshape(conv3_weight.shape[0], -1).mean(dim=1)
    return _max_normalize(per_neuron)


def _infer_label_counts_from_decision_map(
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
class NeuronPartitionConfig:
    enabled: bool = True
    f_stable_percentile: float = 0.70
    f_shared_percentile: float = 0.40
    q_stable_min: float = 0.60
    use_weight_strength: bool = True
    treat_dead_as_reserve: bool = True
    num_classes: Optional[int] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "NeuronPartitionConfig":
        importance_cfg = dict(config.get("importance", {}))
        return cls(
            enabled=bool(config.get("enabled", True)),
            f_stable_percentile=float(config.get("f_stable_percentile", 0.70)),
            f_shared_percentile=float(config.get("f_shared_percentile", 0.40)),
            q_stable_min=float(config.get("q_stable_min", 0.60)),
            use_weight_strength=bool(importance_cfg.get("use_weight_strength", True)),
            treat_dead_as_reserve=bool(config.get("treat_dead_as_reserve", True)),
            num_classes=config.get("num_classes"),
        )


@dataclass
class NeuronPartition:
    """Partition S3 neurons into stable / shared / reserve pools after Task 1."""

    roles: Tensor
    f_i: Tensor
    q_i: Tensor
    I_i: Tensor
    dominant_labels: Tensor
    config: NeuronPartitionConfig
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def num_neurons(self) -> int:
        return int(self.roles.numel())

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NeuronPartition":
        partition_config = NeuronPartitionConfig.from_mapping(config)
        if not partition_config.enabled:
            return cls(
                roles=torch.tensor([], dtype=torch.int64),
                f_i=torch.tensor([]),
                q_i=torch.tensor([]),
                I_i=torch.tensor([]),
                dominant_labels=torch.tensor([], dtype=torch.int64),
                config=partition_config,
            )
        raise ValueError(
            "Enabled NeuronPartition must be built from Task 1 stats via fit_from_task1_stats()."
        )

    @classmethod
    def fit_from_task1_stats(
        cls,
        model: nn.Module,
        winner_counts: Sequence[int],
        config: Mapping[str, Any] | NeuronPartitionConfig,
        *,
        winner_label_counts: Optional[Union[Sequence[Sequence[int]], Tensor]] = None,
        num_classes: Optional[int] = None,
    ) -> "NeuronPartition":
        partition_config = (
            config if isinstance(config, NeuronPartitionConfig) else NeuronPartitionConfig.from_mapping(config)
        )
        if not partition_config.enabled:
            return cls.from_config({"enabled": False})

        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not hasattr(conv3, "weight"):
            raise ValueError("NeuronPartition requires a paper-source model with conv3 weights.")

        resolved_num_classes = num_classes
        if resolved_num_classes is None:
            resolved_num_classes = partition_config.num_classes
        if resolved_num_classes is None:
            model_config = getattr(model, "config", None)
            resolved_num_classes = getattr(model_config, "num_classes", None)
        if resolved_num_classes is None:
            raise ValueError("num_classes is required when winner_label_counts are unavailable.")

        return cls.fit_from_arrays(
            conv3_weight=conv3.weight.detach(),
            winner_counts=winner_counts,
            winner_label_counts=winner_label_counts,
            decision_map=getattr(model, "decision_map", None),
            num_classes=int(resolved_num_classes),
            config=partition_config,
        )

    @classmethod
    def fit_from_arrays(
        cls,
        *,
        conv3_weight: Tensor,
        winner_counts: Sequence[int],
        config: NeuronPartitionConfig | Mapping[str, Any],
        winner_label_counts: Optional[Union[Sequence[Sequence[int]], Tensor]] = None,
        decision_map: Optional[Sequence[int]] = None,
        num_classes: int,
    ) -> "NeuronPartition":
        if not isinstance(config, NeuronPartitionConfig):
            partition_config = NeuronPartitionConfig.from_mapping(config)
        else:
            partition_config = config

        if not partition_config.enabled:
            return cls.from_config({"enabled": False})

        num_neurons = int(conv3_weight.shape[0])
        f_i = _winner_counts_tensor(winner_counts, num_neurons)
        label_counts = _as_2d_label_counts(winner_label_counts, num_neurons, num_classes)
        if float(label_counts.sum().item()) <= 0.0 and decision_map is not None:
            label_counts = _infer_label_counts_from_decision_map(f_i, decision_map, num_classes)

        q_i, dominant_labels = _compute_selectivity(label_counts)
        I_i = _compute_neuron_importance(conv3_weight, partition_config.use_weight_strength)
        roles, thresholds = cls._assign_roles(f_i, q_i, partition_config)
        return cls(
            roles=roles,
            f_i=f_i,
            q_i=q_i,
            I_i=I_i,
            dominant_labels=dominant_labels,
            config=partition_config,
            thresholds=thresholds,
        )

    @classmethod
    def _assign_roles(
        cls,
        f_i: Tensor,
        q_i: Tensor,
        config: NeuronPartitionConfig,
    ) -> Tuple[Tensor, Dict[str, float]]:
        num_neurons = int(f_i.numel())
        roles = torch.full((num_neurons,), int(NeuronRole.RESERVE), dtype=torch.int64)
        f_stable_thr = _percentile_threshold(f_i, config.f_stable_percentile)
        f_shared_thr = _percentile_threshold(f_i, config.f_shared_percentile)
        if f_shared_thr > f_stable_thr:
            f_shared_thr = f_stable_thr

        for neuron_idx in range(num_neurons):
            wins = float(f_i[neuron_idx].item())
            selectivity = float(q_i[neuron_idx].item())
            if wins <= 0:
                roles[neuron_idx] = int(NeuronRole.DEAD if not config.treat_dead_as_reserve else NeuronRole.RESERVE)
                continue
            if wins >= f_stable_thr and selectivity >= config.q_stable_min:
                roles[neuron_idx] = int(NeuronRole.STABLE)
            elif wins >= f_shared_thr:
                roles[neuron_idx] = int(NeuronRole.SHARED)
            else:
                roles[neuron_idx] = int(NeuronRole.RESERVE)

        thresholds = {
            "f_stable_threshold": f_stable_thr,
            "f_shared_threshold": f_shared_thr,
            "q_stable_min": float(config.q_stable_min),
            "f_stable_percentile": float(config.f_stable_percentile),
            "f_shared_percentile": float(config.f_shared_percentile),
        }
        return roles, thresholds

    def role_name(self, neuron_idx: int) -> str:
        role = NeuronRole(int(self.roles[neuron_idx].item()))
        return ROLE_NAMES[role]

    def mask_for_role(self, role: NeuronRole | str) -> Tensor:
        if isinstance(role, str):
            lookup = {name: enum_role for enum_role, name in ROLE_NAMES.items()}
            if role not in lookup:
                raise ValueError(f"Unknown neuron role '{role}'. Expected one of {sorted(lookup)}.")
            role = lookup[role]
        return self.roles == int(role)

    def indices_for_role(self, role: NeuronRole | str) -> Tensor:
        return self.mask_for_role(role).nonzero(as_tuple=False).reshape(-1)

    def combined_score(self) -> Tensor:
        """Neuron-level old-task occupancy score used by SDPM / reserve routing."""
        return _max_normalize(self.f_i) * self.q_i * self.I_i

    def counts_by_role(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in ROLE_NAMES.values()}
        for neuron_idx in range(self.num_neurons):
            counts[self.role_name(neuron_idx)] += 1
        return counts

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled or self.roles.numel() == 0:
            return {"enabled": False}

        role_counts = self.counts_by_role()
        total = max(self.num_neurons, 1)
        return {
            "enabled": True,
            "num_neurons": self.num_neurons,
            "role_counts": role_counts,
            "role_fractions": {role: float(count / total) for role, count in role_counts.items()},
            "thresholds": dict(self.thresholds),
            "f_i_mean": float(self.f_i.mean().item()),
            "f_i_max": float(self.f_i.max().item()),
            "q_i_mean": float(self.q_i[self.f_i > 0].mean().item()) if (self.f_i > 0).any() else 0.0,
            "I_i_mean": float(self.I_i.mean().item()),
            "combined_score_mean": float(self.combined_score().mean().item()),
            "config": asdict(self.config),
        }

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        payload = self.summarize()
        if include_arrays and self.enabled and self.roles.numel() > 0:
            payload.update(
                {
                    "roles": [ROLE_NAMES[NeuronRole(int(role))] for role in self.roles.tolist()],
                    "dominant_labels": self.dominant_labels.tolist(),
                    "f_i": self.f_i.tolist(),
                    "q_i": self.q_i.tolist(),
                    "I_i": self.I_i.tolist(),
                    "combined_score": self.combined_score().tolist(),
                }
            )
        return payload
