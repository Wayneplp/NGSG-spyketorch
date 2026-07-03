from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .neuron_partition import NeuronPartition, NeuronRole, ROLE_NAMES
from .novelty_gate import NoveltyGate, NoveltyGateConfig


def _parse_apply_stages(config: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = config.get("apply_stages", ("task2",))
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(stage) for stage in raw)


def _parse_recruit_roles(config: Mapping[str, Any]) -> Tuple[NeuronRole, ...]:
    raw = config.get("recruit_roles", ("reserve",))
    if isinstance(raw, str):
        raw = (raw,)
    lookup = {name: role for role, name in ROLE_NAMES.items()}
    roles: List[NeuronRole] = []
    for item in raw:
        key = str(item).lower()
        if key not in lookup:
            raise ValueError(f"Unknown recruit role '{item}'. Expected one of {sorted(lookup)}.")
        roles.append(lookup[key])
    return tuple(roles)


def _resolve_num_s3_neurons(model: nn.Module, pot: Optional[Tensor] = None) -> int:
    model_config = getattr(model, "config", None)
    num_neurons = int(getattr(model_config, "s3_neurons", 0) or 0)
    if num_neurons <= 0:
        decision_map = getattr(model, "decision_map", None)
        num_neurons = len(decision_map) if decision_map is not None else 0
    if num_neurons <= 0 and pot is not None and pot.ndim >= 2:
        if pot.ndim == 3:
            num_neurons = int(pot.shape[0])
        elif pot.shape[1] > 0:
            num_neurons = int(pot.shape[1])
    return num_neurons


def potential_planes(pot: Tensor, num_neurons: int) -> Tensor:
    """Return S3 potentials as [num_neurons, H, W]."""
    pot = pot.detach().float().cpu()
    if pot.ndim == 3 and int(pot.shape[0]) == num_neurons:
        return pot
    if pot.ndim == 4 and int(pot.shape[1]) == num_neurons:
        return pot.sum(dim=0)
    if pot.ndim == 4 and int(pot.shape[0]) == num_neurons:
        return pot.sum(dim=1)
    raise ValueError(
        f"Unexpected S3 potential shape {tuple(pot.shape)} for num_neurons={num_neurons}."
    )


def aggregate_s3_neuron_potentials(model: nn.Module) -> Optional[Tensor]:
    ctx = getattr(model, "ctx", None)
    if not isinstance(ctx, dict):
        return None
    pot = ctx.get("potentials")
    if pot is None:
        return None
    num_neurons = _resolve_num_s3_neurons(model, pot)
    if num_neurons <= 0:
        return None
    planes = potential_planes(pot, num_neurons)
    return planes.reshape(num_neurons, -1).sum(dim=1)


def build_winner_entry(pot: Tensor, neuron_idx: int, template_winner: Any, *, num_neurons: int) -> Any:
    plane = potential_planes(pot, num_neurons)[int(neuron_idx)]
    flat = int(plane.argmax().item())
    height, width = int(plane.shape[0]), int(plane.shape[1])
    row, col = divmod(flat, max(width, 1))
    try:
        template_len = len(template_winner)
    except TypeError:
        template_len = 1
    if template_len >= 3:
        return (int(neuron_idx), int(row), int(col))
    if template_len == 2:
        return (int(neuron_idx), float(plane.max().item()))
    return (int(neuron_idx),)


@dataclass
class ReserveActivationConfig:
    enabled: bool = False
    apply_stages: Tuple[str, ...] = ("task2",)
    novelty_threshold: float = 0.15
    use_class_local: bool = True
    recruit_roles: Tuple[NeuronRole, ...] = (NeuronRole.RESERVE,)
    random_recruitment: bool = False
    random_seed: Optional[int] = None
    use_normalized_occupancy: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ReserveActivationConfig":
        return cls(
            enabled=bool(config.get("enabled", False)),
            apply_stages=_parse_apply_stages(config),
            novelty_threshold=float(config.get("novelty_threshold", 0.15)),
            use_class_local=bool(config.get("use_class_local", True)),
            recruit_roles=_parse_recruit_roles(config),
            random_recruitment=bool(config.get("random_recruitment", False)),
            random_seed=config.get("random_seed"),
            use_normalized_occupancy=bool(config.get("use_normalized_occupancy", True)),
        )


@dataclass
class ReserveActivation:
    """Reroute high-novelty Task 2 STDP updates toward low-occupancy neurons."""

    partition: NeuronPartition
    novelty_gate: NoveltyGate
    config: ReserveActivationConfig
    neurons_per_class: int
    stats: Dict[str, float] = field(default_factory=dict)
    _generator: Optional[torch.Generator] = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.partition.enabled and self.novelty_gate.enabled

    @classmethod
    def from_partition(
        cls,
        partition: NeuronPartition,
        config: Mapping[str, Any] | ReserveActivationConfig,
        *,
        neurons_per_class: int,
        global_seed: Optional[int] = None,
    ) -> "ReserveActivation":
        reserve_config = (
            config if isinstance(config, ReserveActivationConfig) else ReserveActivationConfig.from_mapping(config)
        )
        if not reserve_config.enabled or not partition.enabled:
            empty_novelty = NoveltyGate(occupancy=torch.tensor([]), config=NoveltyGateConfig(enabled=False))
            return cls(
                partition=partition,
                novelty_gate=empty_novelty,
                config=reserve_config,
                neurons_per_class=int(neurons_per_class),
            )

        novelty_cfg = {
            "enabled": True,
            "novelty_threshold": reserve_config.novelty_threshold,
            "use_normalized_occupancy": reserve_config.use_normalized_occupancy,
            "apply_stages": list(reserve_config.apply_stages),
        }
        novelty_gate = NoveltyGate.from_partition(partition, novelty_cfg)
        seed = reserve_config.random_seed if reserve_config.random_seed is not None else global_seed
        generator = None
        if reserve_config.random_recruitment and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))

        return cls(
            partition=partition,
            novelty_gate=novelty_gate,
            config=reserve_config,
            neurons_per_class=int(neurons_per_class),
            _generator=generator,
        )

    def should_apply(self, stage_name: str) -> bool:
        return self.enabled and stage_name in self.config.apply_stages

    def _class_block(self, class_idx: int, num_neurons: int) -> range:
        start = int(class_idx) * self.neurons_per_class
        end = min(start + self.neurons_per_class, num_neurons)
        return range(start, end)

    def _candidate_indices(
        self,
        *,
        target_class: int,
        num_neurons: int,
    ) -> Tensor:
        if self.config.use_class_local:
            search_range: Iterable[int] = self._class_block(target_class, num_neurons)
        else:
            search_range = range(num_neurons)

        role_mask = torch.zeros(num_neurons, dtype=torch.bool)
        for role in self.config.recruit_roles:
            role_mask |= self.partition.mask_for_role(role)

        candidates = [idx for idx in search_range if bool(role_mask[idx].item())]
        if not candidates and self.config.use_class_local:
            candidates = [idx for idx in range(num_neurons) if bool(role_mask[idx].item())]
        if not candidates:
            return torch.tensor([], dtype=torch.int64)
        return torch.tensor(candidates, dtype=torch.int64)

    def _select_neuron(
        self,
        *,
        potentials: Tensor,
        target_class: int,
        *,
        num_neurons: int,
    ) -> Optional[int]:
        candidates = self._candidate_indices(target_class=target_class, num_neurons=num_neurons)
        if candidates.numel() == 0:
            return None

        if self.config.random_recruitment:
            pick = int(candidates[torch.randint(candidates.numel(), (1,), generator=self._generator).item()].item())
            return pick

        candidate_potentials = potentials[candidates]
        best_local = int(candidate_potentials.argmax().item())
        return int(candidates[best_local].item())

    def maybe_reroute(
        self,
        model: nn.Module,
        *,
        natural_winner_idx: Optional[int],
        target_class: int,
        stage_name: str,
    ) -> bool:
        if not self.should_apply(stage_name):
            return False

        self.novelty_gate.observe(natural_winner_idx)
        if not self.novelty_gate.is_novel(natural_winner_idx):
            self.stats["skipped_low_novelty"] = float(self.stats.get("skipped_low_novelty", 0.0) + 1.0)
            return False

        potentials = aggregate_s3_neuron_potentials(model)
        ctx = getattr(model, "ctx", None)
        if potentials is None or not isinstance(ctx, dict) or ctx.get("winners") is None:
            return False

        pot = ctx.get("potentials")
        if pot is None:
            return False
        num_neurons = _resolve_num_s3_neurons(model, pot)
        if num_neurons <= 0:
            return False

        recruited = self._select_neuron(
            potentials=potentials,
            target_class=int(target_class),
            num_neurons=num_neurons,
        )
        if recruited is None:
            self.stats["failed_recruitment"] = float(self.stats.get("failed_recruitment", 0.0) + 1.0)
            return False

        natural_winners = ctx["winners"]
        template = natural_winners[0] if len(natural_winners) > 0 else (int(recruited),)
        ctx["winners"] = [
            build_winner_entry(pot, recruited, template, num_neurons=num_neurons)
        ]

        self.stats["recruited_updates"] = float(self.stats.get("recruited_updates", 0.0) + 1.0)
        return True

    def summarize(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}

        recruited = float(self.stats.get("recruited_updates", 0.0))
        samples = float(self.novelty_gate.stats.get("samples", 0.0))
        return {
            "enabled": True,
            "recruited_updates": recruited,
            "recruitment_rate": float(recruited / samples) if samples > 0 else 0.0,
            "failed_recruitment": float(self.stats.get("failed_recruitment", 0.0)),
            "skipped_low_novelty": float(self.stats.get("skipped_low_novelty", 0.0)),
            "random_recruitment": self.config.random_recruitment,
            "recruit_roles": [ROLE_NAMES[role] for role in self.config.recruit_roles],
            "config": asdict(self.config),
            "novelty_gate": self.novelty_gate.summarize(),
        }
