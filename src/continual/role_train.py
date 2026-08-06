from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .neuron_partition import NeuronPartition, NeuronRole


def winner_distribution_metrics(counts: Sequence[int]) -> Dict[str, float]:
    """Summarize WTA winner concentration from per-neuron win counts."""
    total = float(sum(int(c) for c in counts))
    if total <= 0.0:
        return {
            "unique_winners": 0.0,
            "top1_winner_share": 0.0,
            "top5_winner_share": 0.0,
            "winner_entropy": 0.0,
            "winner_entropy_normalized": 0.0,
            "total_winner_events": 0.0,
        }

    sorted_counts = sorted((int(c) for c in counts), reverse=True)
    unique = sum(1 for count in sorted_counts if count > 0)
    top1 = sorted_counts[0] / total
    top5 = sum(sorted_counts[:5]) / total
    entropy = 0.0
    for count in sorted_counts:
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    max_entropy = math.log(unique) if unique > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0.0 else 0.0
    return {
        "unique_winners": float(unique),
        "top1_winner_share": float(top1),
        "top5_winner_share": float(top5),
        "winner_entropy": float(entropy),
        "winner_entropy_normalized": float(normalized),
        "total_winner_events": total,
    }


@dataclass
class RoleTrainCompetitionTracker:
    """Task2 role-train competition / credit-assignment diagnostics."""

    num_neurons: int
    winner_counts: List[int] = field(default_factory=list)
    forward_with_winner: int = 0
    forward_silent: int = 0
    stdp_eligible: int = 0
    effective_stdp_updates: int = 0
    gated_skipped_stdp: int = 0
    updates_by_role: Dict[str, int] = field(default_factory=dict)
    epoch_history: List[Dict[str, Any]] = field(default_factory=list)
    _epoch_winner_counts: List[int] = field(default_factory=list, repr=False)
    _epoch_forward_with_winner: int = field(default=0, repr=False)
    _epoch_forward_silent: int = field(default=0, repr=False)
    _epoch_stdp_eligible: int = field(default=0, repr=False)
    _epoch_effective_stdp_updates: int = field(default=0, repr=False)
    _epoch_gated_skipped_stdp: int = field(default=0, repr=False)
    _epoch_updates_by_role: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.winner_counts:
            self.winner_counts = [0] * int(self.num_neurons)
        if not self.updates_by_role:
            self.updates_by_role = self._empty_role_counter()
        self._reset_epoch()

    @staticmethod
    def _empty_role_counter() -> Dict[str, int]:
        return {"stable": 0, "shared": 0, "reserve": 0, "dead": 0, "unknown": 0}

    def _reset_epoch(self) -> None:
        self._epoch_winner_counts = [0] * int(self.num_neurons)
        self._epoch_forward_with_winner = 0
        self._epoch_forward_silent = 0
        self._epoch_stdp_eligible = 0
        self._epoch_effective_stdp_updates = 0
        self._epoch_gated_skipped_stdp = 0
        self._epoch_updates_by_role = self._empty_role_counter()

    def begin_epoch(self) -> None:
        self._reset_epoch()

    def record_sample(
        self,
        *,
        winner_idx: Optional[int],
        forward_silent: bool,
        stdp_eligible: bool,
        multiplier: float,
        update_applied: bool,
        role_name: str,
    ) -> None:
        if forward_silent:
            self.forward_silent += 1
            self._epoch_forward_silent += 1
        elif winner_idx is not None:
            self.forward_with_winner += 1
            self._epoch_forward_with_winner += 1
            if 0 <= winner_idx < self.num_neurons:
                self.winner_counts[winner_idx] += 1
                self._epoch_winner_counts[winner_idx] += 1

        if not stdp_eligible:
            return

        self.stdp_eligible += 1
        self._epoch_stdp_eligible += 1
        if update_applied:
            self.effective_stdp_updates += 1
            self._epoch_effective_stdp_updates += 1
            bucket = role_name if role_name in self.updates_by_role else "unknown"
            self.updates_by_role[bucket] = self.updates_by_role.get(bucket, 0) + 1
            self._epoch_updates_by_role[bucket] = self._epoch_updates_by_role.get(bucket, 0) + 1
            return

        if winner_idx is not None and float(multiplier) == 0.0:
            self.gated_skipped_stdp += 1
            self._epoch_gated_skipped_stdp += 1

    def _update_share_summary(
        self,
        *,
        forward_with_winner: int,
        stdp_eligible: int,
        effective_stdp_updates: int,
        gated_skipped_stdp: int,
        updates_by_role: Mapping[str, int],
    ) -> Dict[str, Any]:
        forward_winners = max(int(forward_with_winner), 0)
        eligible = max(int(stdp_eligible), 0)
        effective = max(int(effective_stdp_updates), 0)
        gated = max(int(gated_skipped_stdp), 0)
        shared_updates = int(updates_by_role.get("shared", 0))
        reserve_updates = int(updates_by_role.get("reserve", 0))
        open_pool_updates = shared_updates + reserve_updates
        role_update_share = {
            role: float(int(updates_by_role.get(role, 0)) / effective) if effective > 0 else 0.0
            for role in ("stable", "shared", "reserve", "dead", "unknown")
        }
        return {
            "stdp_eligible_samples": float(eligible),
            "effective_stdp_updates": float(effective),
            "gated_skipped_stdp_samples": float(gated),
            "forward_winner_gated_no_update_samples": float(gated),
            "has_forward_winner_gated_no_update": bool(gated > 0),
            "forward_winner_gated_no_update_rate": (
                float(gated / forward_winners) if forward_winners > 0 else 0.0
            ),
            "gated_skip_rate": float(gated / eligible) if eligible > 0 else 0.0,
            "effective_update_rate": float(effective / eligible) if eligible > 0 else 0.0,
            "updates_by_role": dict(updates_by_role),
            "role_update_share": role_update_share,
            "shared_update_share": float(shared_updates / effective) if effective > 0 else 0.0,
            "reserve_update_share": float(reserve_updates / effective) if effective > 0 else 0.0,
            "open_pool_update_share": {
                "shared": float(shared_updates / open_pool_updates) if open_pool_updates > 0 else 0.0,
                "reserve": float(reserve_updates / open_pool_updates) if open_pool_updates > 0 else 0.0,
            },
        }

    def end_epoch(self, epoch_index: int) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "epoch": int(epoch_index) + 1,
            **winner_distribution_metrics(self._epoch_winner_counts),
            "forward_with_winner": float(self._epoch_forward_with_winner),
            "forward_silent": float(self._epoch_forward_silent),
            **self._update_share_summary(
                forward_with_winner=self._epoch_forward_with_winner,
                stdp_eligible=self._epoch_stdp_eligible,
                effective_stdp_updates=self._epoch_effective_stdp_updates,
                gated_skipped_stdp=self._epoch_gated_skipped_stdp,
                updates_by_role=self._epoch_updates_by_role,
            ),
        }
        self.epoch_history.append(summary)
        return summary

    def stage_summary(self) -> Dict[str, Any]:
        distribution = winner_distribution_metrics(self.winner_counts)
        updates = self._update_share_summary(
            forward_with_winner=self.forward_with_winner,
            stdp_eligible=self.stdp_eligible,
            effective_stdp_updates=self.effective_stdp_updates,
            gated_skipped_stdp=self.gated_skipped_stdp,
            updates_by_role=self.updates_by_role,
        )
        return {
            **distribution,
            "forward_with_winner": float(self.forward_with_winner),
            "forward_silent": float(self.forward_silent),
            **updates,
            "epoch_history": list(self.epoch_history),
        }


@dataclass
class RolePhaseConfig:
    stable_wta_open: bool = False
    shared_wta_open: bool = True
    reserve_wta_open: bool = True
    stable_stdp_lr: float = 0.0
    shared_stdp_lr: float = 0.2
    reserve_stdp_lr: float = 1.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "RolePhaseConfig":
        def role_block(name: str, defaults: Mapping[str, Any]) -> Dict[str, Any]:
            block = dict(config.get(name, {}))
            return {key: block.get(key, defaults[key]) for key in defaults}

        stable = role_block("stable", {"wta_open": False, "stdp_lr": 0.0})
        shared = role_block("shared", {"wta_open": True, "stdp_lr": 0.2})
        reserve = role_block("reserve", {"wta_open": True, "stdp_lr": 1.0})
        return cls(
            stable_wta_open=bool(stable["wta_open"]),
            shared_wta_open=bool(shared["wta_open"]),
            reserve_wta_open=bool(reserve["wta_open"]),
            stable_stdp_lr=float(stable["stdp_lr"]),
            shared_stdp_lr=float(shared["stdp_lr"]),
            reserve_stdp_lr=float(reserve["stdp_lr"]),
        )


@dataclass
class RoleTrainConfig:
    enabled: bool = False
    apply_stages: Sequence[str] = field(default_factory=lambda: ("task2",))
    early_epoch_fraction: float = 0.8
    early: RolePhaseConfig = field(default_factory=RolePhaseConfig)
    late: RolePhaseConfig = field(default_factory=lambda: RolePhaseConfig(
        stable_wta_open=False,
        shared_wta_open=True,
        reserve_wta_open=True,
        stable_stdp_lr=0.0,
        shared_stdp_lr=0.1,
        reserve_stdp_lr=0.5,
    ))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "RoleTrainConfig":
        return cls(
            enabled=bool(config.get("enabled", False)),
            apply_stages=tuple(config.get("apply_stages", ["task2"])),
            early_epoch_fraction=float(config.get("early_epoch_fraction", 0.8)),
            early=RolePhaseConfig.from_mapping(config.get("early", {})),
            late=RolePhaseConfig.from_mapping(config.get("late", {})),
        )


@dataclass
class RoleTrainSchedule:
    """Task2 role-aware WTA mask schedule and S3 STDP lr multipliers."""

    config: RoleTrainConfig
    partition: NeuronPartition
    stdp_multipliers: Tensor = field(default_factory=lambda: torch.tensor([]))
    stats: Dict[str, float] = field(default_factory=dict)
    competition: Optional[RoleTrainCompetitionTracker] = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.partition.enabled

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        partition: NeuronPartition,
    ) -> "RoleTrainSchedule":
        role_config = RoleTrainConfig.from_mapping(config.get("role_train", {}))
        schedule = cls(config=role_config, partition=partition)
        if schedule.enabled:
            schedule.stdp_multipliers = schedule._build_stdp_multipliers(role_config.early)
            schedule.competition = RoleTrainCompetitionTracker(num_neurons=partition.num_neurons)
        return schedule

    def should_apply(self, stage_name: str) -> bool:
        return self.enabled and stage_name in self.config.apply_stages

    def _phase_for_epoch(self, epoch_index: int, total_epochs: int) -> RolePhaseConfig:
        if total_epochs <= 0:
            return self.config.early
        cutoff = int(total_epochs * float(self.config.early_epoch_fraction))
        cutoff = min(max(cutoff, 1), total_epochs)
        if epoch_index < cutoff:
            return self.config.early
        return self.config.late

    def _role_wta_open(self, phase: RolePhaseConfig, role: NeuronRole | str) -> bool:
        if isinstance(role, str):
            lookup = {
                "stable": phase.stable_wta_open,
                "shared": phase.shared_wta_open,
                "reserve": phase.reserve_wta_open,
            }
            return bool(lookup[role])
        mapping = {
            NeuronRole.STABLE: phase.stable_wta_open,
            NeuronRole.SHARED: phase.shared_wta_open,
            NeuronRole.RESERVE: phase.reserve_wta_open,
        }
        return bool(mapping.get(role, False))

    def wta_allow_mask(self, epoch_index: int, total_epochs: int) -> Tensor:
        phase = self._phase_for_epoch(epoch_index, total_epochs)
        allow = torch.zeros((self.partition.num_neurons,), dtype=torch.bool)
        for role_name in ("stable", "shared", "reserve"):
            if self._role_wta_open(phase, role_name):
                allow |= self.partition.mask_for_role(role_name)
        return allow

    def _build_stdp_multipliers(self, phase: RolePhaseConfig) -> Tensor:
        multipliers = torch.zeros((self.partition.num_neurons,), dtype=torch.float32)
        role_values = {
            "stable": float(phase.stable_stdp_lr),
            "shared": float(phase.shared_stdp_lr),
            "reserve": float(phase.reserve_stdp_lr),
        }
        for role_name, value in role_values.items():
            multipliers[self.partition.mask_for_role(role_name)] = value
        dead_mask = self.partition.mask_for_role("dead")
        if bool(dead_mask.any()):
            multipliers[dead_mask] = 0.0
        return multipliers

    def set_epoch(self, epoch_index: int, total_epochs: int) -> None:
        phase = self._phase_for_epoch(epoch_index, total_epochs)
        self.stdp_multipliers = self._build_stdp_multipliers(phase)

    def apply_delta(self, weight_before: Tensor, weight_after: Tensor) -> Tensor:
        if not self.enabled or self.stdp_multipliers.numel() == 0:
            return weight_after
        delta = weight_after - weight_before
        multipliers = self.stdp_multipliers.to(device=delta.device, dtype=delta.dtype).view(-1, 1, 1, 1)
        if multipliers.shape[0] != delta.shape[0]:
            raise ValueError("RoleTrain STDP multipliers must match conv3 out_channels.")
        return weight_before + multipliers * delta

    def apply_to_conv3(self, model: nn.Module, weight_before: Tensor, weight_after: Tensor) -> None:
        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not hasattr(conv3, "weight"):
            raise ValueError("RoleTrainSchedule.apply_to_conv3 requires model.conv3.weight.")
        conv3.weight.copy_(self.apply_delta(weight_before, weight_after))

    def multiplier_for_winner(self, model: nn.Module) -> float:
        winners = getattr(model, "ctx", {}).get("winners") if hasattr(model, "ctx") else None
        if winners is None or len(winners) == 0 or self.stdp_multipliers.numel() == 0:
            return 1.0
        try:
            winner_idx = int(winners[0][0])
        except (TypeError, ValueError, IndexError):
            return 1.0
        if winner_idx < 0 or winner_idx >= int(self.stdp_multipliers.numel()):
            return 1.0
        return float(self.stdp_multipliers[winner_idx].item())

    def role_name_for_winner(self, winner_idx: Optional[int]) -> str:
        if winner_idx is None:
            return "unknown"
        try:
            return str(self.partition.role_name(int(winner_idx)))
        except (TypeError, ValueError, IndexError):
            return "unknown"

    @contextmanager
    def _temporarily_scaled_learning_rate(self, stdp: Any, multiplier: float):
        old_ap = float(stdp.learning_rate[0][0].item())
        old_an = float(stdp.learning_rate[0][1].item())
        stdp.update_all_learning_rate(old_ap * multiplier, old_an * multiplier)
        try:
            yield
        finally:
            stdp.update_all_learning_rate(old_ap, old_an)

    @torch.no_grad()
    def gated_reward(self, model: nn.Module) -> bool:
        if not hasattr(model, "reward"):
            raise ValueError("Model does not expose reward().")
        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not self.enabled:
            model.reward()
            return True
        multiplier = self.multiplier_for_winner(model)
        if multiplier == 0.0:
            self.stats["reward_skipped"] = float(self.stats.get("reward_skipped", 0.0) + 1.0)
            return False
        stdp = getattr(model, "stdp3", None)
        if stdp is None:
            weight_before = conv3.weight.clone()
            model.reward()
            self.apply_to_conv3(model, weight_before, conv3.weight)
        elif multiplier == 1.0:
            model.reward()
        else:
            with self._temporarily_scaled_learning_rate(stdp, multiplier):
                model.reward()
        self.stats["reward_calls"] = float(self.stats.get("reward_calls", 0.0) + 1.0)
        return True

    @torch.no_grad()
    def gated_punish(self, model: nn.Module) -> bool:
        if not hasattr(model, "punish"):
            raise ValueError("Model does not expose punish().")
        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not self.enabled:
            model.punish()
            return True
        multiplier = self.multiplier_for_winner(model)
        if multiplier == 0.0:
            self.stats["punish_skipped"] = float(self.stats.get("punish_skipped", 0.0) + 1.0)
            return False
        stdp = getattr(model, "anti_stdp3", None)
        if stdp is None:
            weight_before = conv3.weight.clone()
            model.punish()
            self.apply_to_conv3(model, weight_before, conv3.weight)
        elif multiplier == 1.0:
            model.punish()
        else:
            with self._temporarily_scaled_learning_rate(stdp, multiplier):
                model.punish()
        self.stats["punish_calls"] = float(self.stats.get("punish_calls", 0.0) + 1.0)
        return True

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        multipliers = self.stdp_multipliers
        return {
            "enabled": True,
            "apply_stages": list(self.config.apply_stages),
            "early_epoch_fraction": self.config.early_epoch_fraction,
            "early": asdict(self.config.early),
            "late": asdict(self.config.late),
            "stdp_multiplier_mean": float(multipliers.mean().item()) if multipliers.numel() else 0.0,
            "stdp_multiplier_by_role": {
                role: float(multipliers[self.partition.mask_for_role(role)].mean().item())
                if bool(self.partition.mask_for_role(role).any())
                else 0.0
                for role in ("stable", "shared", "reserve")
            },
            "stats": dict(self.stats),
            "competition_diagnostics": (
                self.competition.stage_summary() if self.competition is not None else None
            ),
        }
