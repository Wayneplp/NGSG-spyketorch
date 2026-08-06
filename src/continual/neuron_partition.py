from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .occupancy_stats import (
    Task1OccupancyStats,
    fit_task1_occupancy_stats,
)


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


def _percentile_threshold(values: Tensor, percentile: float) -> float:
    active = values[values > 0]
    if active.numel() == 0:
        return 0.0
    percentile = float(min(max(percentile, 0.0), 1.0))
    return float(torch.quantile(active, percentile).item())


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
    occupancy: Task1OccupancyStats
    config: NeuronPartitionConfig
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def num_neurons(self) -> int:
        return int(self.roles.numel())

    @property
    def f_i(self) -> Tensor:
        return self.occupancy.f_i

    @property
    def q_i(self) -> Tensor:
        return self.occupancy.q_i

    @property
    def I_i(self) -> Tensor:
        return self.occupancy.I_i

    @property
    def dominant_labels(self) -> Tensor:
        return self.occupancy.dominant_labels

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NeuronPartition":
        partition_config = NeuronPartitionConfig.from_mapping(config)
        if not partition_config.enabled:
            empty = Task1OccupancyStats(
                f_i=torch.tensor([]),
                q_i=torch.tensor([]),
                I_i=torch.tensor([]),
                dominant_labels=torch.tensor([], dtype=torch.int64),
            )
            return cls(
                roles=torch.tensor([], dtype=torch.int64),
                occupancy=empty,
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

        occupancy = fit_task1_occupancy_stats(
            conv3_weight=conv3_weight,
            winner_counts=winner_counts,
            winner_label_counts=winner_label_counts,
            num_classes=num_classes,
            decision_map=decision_map,
            use_weight_strength=partition_config.use_weight_strength,
            allow_decision_map_fallback=True,
        )
        roles, thresholds = cls._assign_roles(occupancy.f_i, occupancy.q_i, partition_config)
        return cls(
            roles=roles,
            occupancy=occupancy,
            config=partition_config,
            thresholds=thresholds,
        )

    @classmethod
    def fit_from_occupancy(
        cls,
        occupancy: Task1OccupancyStats,
        config: NeuronPartitionConfig | Mapping[str, Any],
    ) -> "NeuronPartition":
        if not isinstance(config, NeuronPartitionConfig):
            partition_config = NeuronPartitionConfig.from_mapping(config)
        else:
            partition_config = config

        if not partition_config.enabled:
            return cls.from_config({"enabled": False})

        roles, thresholds = cls._assign_roles(occupancy.f_i, occupancy.q_i, partition_config)
        return cls(
            roles=roles,
            occupancy=occupancy,
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
        return self.occupancy.combined_score()

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

    @classmethod
    def from_role_payload(cls, payload: Mapping[str, Any]) -> "NeuronPartition":
        """Rebuild a partition from saved role names (e.g. result.json task1_training)."""
        roles_list = payload.get("roles")
        if not roles_list:
            raise ValueError("NeuronPartition.from_role_payload requires a non-empty 'roles' list.")

        name_to_role = {name: int(role) for role, name in ROLE_NAMES.items()}
        roles = torch.tensor([name_to_role[str(name)] for name in roles_list], dtype=torch.int64)
        num_neurons = int(roles.numel())
        config_payload = payload.get("config", {"enabled": True})
        partition_config = (
            config_payload
            if isinstance(config_payload, NeuronPartitionConfig)
            else NeuronPartitionConfig.from_mapping(config_payload)
        )
        occupancy = Task1OccupancyStats(
            f_i=torch.tensor(payload.get("f_i", [0.0] * num_neurons), dtype=torch.float32),
            q_i=torch.tensor(payload.get("q_i", [0.0] * num_neurons), dtype=torch.float32),
            I_i=torch.tensor(payload.get("I_i", [0.0] * num_neurons), dtype=torch.float32),
            dominant_labels=torch.tensor(
                payload.get("dominant_labels", [0] * num_neurons),
                dtype=torch.int64,
            ),
        )
        return cls(
            roles=roles,
            occupancy=occupancy,
            config=partition_config,
            thresholds=dict(payload.get("thresholds", {})),
        )
