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


RECRUIT_CONDITIONS = ("occupancy", "stable_mismatch", "all")


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
    recruit_condition: str = "occupancy"
    update_decision_map: bool = False
    decision_map_update_threshold: int = 3
    reroute_updates: bool = True
    homeostatic_boost: bool = False
    homeostatic_boost_initial: float = 0.0
    homeostatic_boost_start_epoch: int = 1
    homeostatic_boost_decay_epochs: int = 0
    homeostatic_boost_roles: Tuple[NeuronRole, ...] = (NeuronRole.RESERVE,)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ReserveActivationConfig":
        if "recruit_condition" in config:
            recruit_condition = str(config.get("recruit_condition", "occupancy")).lower()
        elif bool(config.get("reroute_only_stable_winner", False)) and bool(
            config.get("reroute_only_on_mismatch", False)
        ):
            recruit_condition = "stable_mismatch"
        else:
            recruit_condition = "occupancy"
        if recruit_condition not in RECRUIT_CONDITIONS:
            raise ValueError(
                f"Unknown reserve recruit_condition '{recruit_condition}'. "
                f"Expected one of {RECRUIT_CONDITIONS}."
            )
        return cls(
            enabled=bool(config.get("enabled", False)),
            apply_stages=_parse_apply_stages(config),
            novelty_threshold=float(config.get("novelty_threshold", 0.15)),
            use_class_local=bool(config.get("use_class_local", True)),
            recruit_roles=_parse_recruit_roles(config),
            random_recruitment=bool(config.get("random_recruitment", False)),
            random_seed=config.get("random_seed"),
            use_normalized_occupancy=bool(config.get("use_normalized_occupancy", True)),
            recruit_condition=recruit_condition,
            update_decision_map=bool(config.get("update_decision_map", False)),
            decision_map_update_threshold=max(1, int(config.get("decision_map_update_threshold", 3))),
            reroute_updates=bool(config.get("reroute_updates", True)),
            homeostatic_boost=bool(config.get("homeostatic_boost", False)),
            homeostatic_boost_initial=float(config.get("homeostatic_boost_initial", 0.0)),
            homeostatic_boost_start_epoch=max(1, int(config.get("homeostatic_boost_start_epoch", 1))),
            homeostatic_boost_decay_epochs=max(0, int(config.get("homeostatic_boost_decay_epochs", 0))),
            homeostatic_boost_roles=_parse_recruit_roles(
                {"recruit_roles": config.get("homeostatic_boost_roles", config.get("recruit_roles", ("reserve",)))}
            ),
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
    _decision_map_recruit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)

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

    def uses_reroute(self) -> bool:
        return self.enabled and self.config.reroute_updates

    def uses_homeostatic_boost(self) -> bool:
        return (
            self.enabled
            and self.config.homeostatic_boost
            and float(self.config.homeostatic_boost_initial) > 0.0
        )

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

    def homeostatic_boost_scale(self, epoch_index: int) -> float:
        if not self.uses_homeostatic_boost():
            return 0.0
        epoch = int(epoch_index) + 1
        start = int(self.config.homeostatic_boost_start_epoch)
        if epoch < start:
            return 0.0
        initial = float(self.config.homeostatic_boost_initial)
        decay_epochs = int(self.config.homeostatic_boost_decay_epochs)
        if decay_epochs <= 0:
            return initial
        step = epoch - start
        if step >= decay_epochs:
            return 0.0
        return initial * float(decay_epochs - step) / float(decay_epochs)

    def homeostatic_boost_vector(self, *, num_neurons: int, epoch_index: int) -> Optional[Tensor]:
        scale = self.homeostatic_boost_scale(epoch_index)
        if scale <= 0.0 or num_neurons <= 0:
            return None

        role_mask = torch.zeros(num_neurons, dtype=torch.bool)
        for role in self.config.homeostatic_boost_roles:
            role_mask |= self.partition.mask_for_role(role)
        if int(role_mask.sum().item()) <= 0:
            return None

        boost = torch.zeros(num_neurons, dtype=torch.float32)
        boost[role_mask] = float(scale)
        self.stats["homeostatic_boost_epochs"] = float(self.stats.get("homeostatic_boost_epochs", 0.0) + 1.0)
        self.stats["homeostatic_boost_last_scale"] = float(scale)
        self.stats["homeostatic_boost_neurons"] = float(role_mask.sum().item())
        return boost

    def _select_neuron(
        self,
        *,
        potentials: Tensor,
        target_class: int,
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

    def _stable_mismatch(self, natural_winner_idx: Optional[int], decision: Optional[int], target_class: int) -> bool:
        if natural_winner_idx is None:
            return False
        if decision is None or int(decision) == int(target_class):
            return False
        return self.partition.role_name(int(natural_winner_idx)) == ROLE_NAMES[NeuronRole.STABLE]

    def _should_recruit(
        self,
        *,
        natural_winner_idx: Optional[int],
        decision: Optional[int],
        target_class: int,
    ) -> Tuple[bool, str]:
        condition = self.config.recruit_condition
        occupancy_novel = self.novelty_gate.is_novel(natural_winner_idx)
        stable_mismatch = self._stable_mismatch(natural_winner_idx, decision, target_class)

        if condition == "occupancy":
            if occupancy_novel:
                return True, "occupancy"
            return False, "skipped_low_novelty"
        if condition == "stable_mismatch":
            if stable_mismatch:
                return True, "stable_mismatch"
            if natural_winner_idx is None:
                return False, "skipped_no_winner"
            if decision is not None and int(decision) == int(target_class):
                return False, "skipped_decision_match"
            if self.partition.role_name(int(natural_winner_idx)) != ROLE_NAMES[NeuronRole.STABLE]:
                return False, "skipped_non_stable_winner"
            return False, "skipped_stable_mismatch"
        if condition == "all":
            if occupancy_novel and stable_mismatch:
                return True, "occupancy_and_stable_mismatch"
            if not occupancy_novel:
                return False, "skipped_low_novelty"
            return False, "skipped_stable_mismatch"
        return False, "skipped_unknown_condition"

    def maybe_reroute(
        self,
        model: nn.Module,
        *,
        natural_winner_idx: Optional[int],
        target_class: int,
        stage_name: str,
        decision: Optional[int] = None,
    ) -> bool:
        if not self.should_apply(stage_name) or not self.config.reroute_updates:
            return False

        self.novelty_gate.observe(natural_winner_idx)
        should_recruit, skip_reason = self._should_recruit(
            natural_winner_idx=natural_winner_idx,
            decision=decision,
            target_class=target_class,
        )
        if not should_recruit:
            self.stats[skip_reason] = float(self.stats.get(skip_reason, 0.0) + 1.0)
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
        recruit_key = f"recruited_by_{skip_reason}"
        self.stats[recruit_key] = float(self.stats.get(recruit_key, 0.0) + 1.0)
        self._maybe_update_decision_map(model, recruited, int(target_class))
        return True

    def _maybe_update_decision_map(self, model: nn.Module, neuron_idx: int, target_class: int) -> None:
        if not self.config.update_decision_map:
            return

        decision_map = getattr(model, "decision_map", None)
        if decision_map is None or neuron_idx < 0 or neuron_idx >= len(decision_map):
            self.stats["decision_map_update_unavailable"] = float(
                self.stats.get("decision_map_update_unavailable", 0.0) + 1.0
            )
            return

        key = (int(neuron_idx), int(target_class))
        count = int(self._decision_map_recruit_counts.get(key, 0) + 1)
        self._decision_map_recruit_counts[key] = count
        self.stats["decision_map_update_observations"] = float(
            self.stats.get("decision_map_update_observations", 0.0) + 1.0
        )

        if count < int(self.config.decision_map_update_threshold):
            return

        current_class = int(decision_map[int(neuron_idx)])
        if current_class == int(target_class):
            self.stats["decision_map_already_aligned"] = float(
                self.stats.get("decision_map_already_aligned", 0.0) + 1.0
            )
            return

        decision_map[int(neuron_idx)] = int(target_class)
        self.stats["decision_map_updates"] = float(self.stats.get("decision_map_updates", 0.0) + 1.0)
        update_key = f"decision_map_{current_class}_to_{int(target_class)}"
        self.stats[update_key] = float(self.stats.get(update_key, 0.0) + 1.0)

    def summarize(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}

        recruited = float(self.stats.get("recruited_updates", 0.0))
        samples = float(self.novelty_gate.stats.get("samples", 0.0))
        skip_keys = (
            "skipped_low_novelty",
            "skipped_no_winner",
            "skipped_decision_match",
            "skipped_non_stable_winner",
            "skipped_stable_mismatch",
            "skipped_unknown_condition",
        )
        skip_stats = {key: float(self.stats.get(key, 0.0)) for key in skip_keys if self.stats.get(key, 0.0)}
        recruit_by = {
            key.removeprefix("recruited_by_"): float(value)
            for key, value in self.stats.items()
            if key.startswith("recruited_by_")
        }
        decision_map_transitions = {
            key.removeprefix("decision_map_"): float(value)
            for key, value in self.stats.items()
            if key.startswith("decision_map_") and "_to_" in key
        }
        return {
            "enabled": True,
            "recruited_updates": recruited,
            "recruitment_rate": float(recruited / samples) if samples > 0 else 0.0,
            "failed_recruitment": float(self.stats.get("failed_recruitment", 0.0)),
            "skip_stats": skip_stats,
            "skipped_low_novelty": float(self.stats.get("skipped_low_novelty", 0.0)),
            "recruited_by": recruit_by,
            "random_recruitment": self.config.random_recruitment,
            "recruit_condition": self.config.recruit_condition,
            "recruit_roles": [ROLE_NAMES[role] for role in self.config.recruit_roles],
            "decision_map_update": {
                "enabled": self.config.update_decision_map,
                "threshold": self.config.decision_map_update_threshold,
                "observations": float(self.stats.get("decision_map_update_observations", 0.0)),
                "updates": float(self.stats.get("decision_map_updates", 0.0)),
                "already_aligned": float(self.stats.get("decision_map_already_aligned", 0.0)),
                "unavailable": float(self.stats.get("decision_map_update_unavailable", 0.0)),
                "transitions": decision_map_transitions,
            },
            "homeostatic_boost": {
                "enabled": self.config.homeostatic_boost,
                "initial": self.config.homeostatic_boost_initial,
                "start_epoch": self.config.homeostatic_boost_start_epoch,
                "decay_epochs": self.config.homeostatic_boost_decay_epochs,
                "roles": [ROLE_NAMES[role] for role in self.config.homeostatic_boost_roles],
                "active_epochs": float(self.stats.get("homeostatic_boost_epochs", 0.0)),
                "last_scale": float(self.stats.get("homeostatic_boost_last_scale", 0.0)),
                "boosted_neurons": float(self.stats.get("homeostatic_boost_neurons", 0.0)),
            },
            "config": asdict(self.config),
            "novelty_gate": self.novelty_gate.summarize(),
        }
