from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


def _max_normalize(values: Tensor, eps: float = 1e-8) -> Tensor:
    values = values.clamp(min=0.0)
    peak = float(values.max().item()) if values.numel() else 0.0
    if peak <= eps:
        return torch.zeros_like(values)
    return values / peak


def _parse_apply_stages(config: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = config.get("apply_stages", ("task2",))
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(stage) for stage in raw)


@dataclass
class SDPMGateConfig:
    enabled: bool = False
    g_min: float = 0.05
    gamma: float = 1.0
    protect_top_fraction: float = 0.3
    use_winner_frequency: bool = True
    use_weight_strength: bool = True
    random_protection: bool = False
    apply_stages: Tuple[str, ...] = ("task2",)
    source_stage: str = "task1"
    track_drift: bool = True
    random_seed: Optional[int] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SDPMGateConfig":
        importance_cfg = dict(config.get("importance", {}))
        return cls(
            enabled=bool(config.get("enabled", False)),
            g_min=float(config.get("g_min", 0.05)),
            gamma=float(config.get("gamma", 1.0)),
            protect_top_fraction=float(config.get("protect_top_fraction", 0.3)),
            use_winner_frequency=bool(importance_cfg.get("use_winner_frequency", True)),
            use_weight_strength=bool(importance_cfg.get("use_weight_strength", True)),
            random_protection=bool(config.get("random_protection", False)),
            apply_stages=_parse_apply_stages(config),
            source_stage=str(config.get("source_stage", "task1")),
            track_drift=bool(config.get("track_drift", True)),
            random_seed=config.get("random_seed"),
        )


@dataclass
class SDPMGate:
    """Soft synaptic protection gate for S3 R-STDP updates."""

    importance: Tensor
    gate: Tensor
    config: SDPMGateConfig
    drift_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SDPMGate":
        gate_config = SDPMGateConfig.from_mapping(config)
        if not gate_config.enabled:
            return cls(
                importance=torch.tensor([]),
                gate=torch.tensor([]),
                config=gate_config,
            )
        raise ValueError("Enabled SDPMGate must be built from Task 1 stats via fit_from_task1_stats().")

    @classmethod
    def fit_from_task1_stats(
        cls,
        model: nn.Module,
        winner_counts: Sequence[int],
        config: Mapping[str, Any],
        *,
        global_seed: Optional[int] = None,
    ) -> "SDPMGate":
        gate_config = SDPMGateConfig.from_mapping(config)
        if not gate_config.enabled:
            return cls(
                importance=torch.tensor([]),
                gate=torch.tensor([]),
                config=gate_config,
            )

        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not hasattr(conv3, "weight"):
            raise ValueError("SDPMGate requires a paper-source model with conv3 weights.")

        seed = gate_config.random_seed if gate_config.random_seed is not None else global_seed
        return cls.fit_from_weights(
            conv3_weight=conv3.weight.detach(),
            winner_counts=winner_counts,
            config=gate_config,
            random_seed=seed,
        )

    @classmethod
    def fit_from_weights(
        cls,
        conv3_weight: Tensor,
        winner_counts: Sequence[int],
        config: SDPMGateConfig | Mapping[str, Any],
        *,
        random_seed: Optional[int] = None,
    ) -> "SDPMGate":
        if not isinstance(config, SDPMGateConfig):
            gate_config = SDPMGateConfig.from_mapping(config)
        else:
            gate_config = config

        if not gate_config.enabled:
            return cls(
                importance=torch.tensor([]),
                gate=torch.tensor([]),
                config=gate_config,
            )

        weight = conv3_weight.detach().float().cpu()
        num_neurons = int(weight.shape[0])
        counts = [int(winner_counts[idx]) if idx < len(winner_counts) else 0 for idx in range(num_neurons)]

        importance = cls._build_importance(weight, counts, gate_config, random_seed=random_seed)
        gate = cls._importance_to_gate(importance, gate_config)
        return cls(importance=importance, gate=gate, config=gate_config)

    @classmethod
    def _build_importance(
        cls,
        conv3_weight: Tensor,
        winner_counts: Sequence[int],
        config: SDPMGateConfig,
        *,
        random_seed: Optional[int] = None,
    ) -> Tensor:
        if config.random_protection:
            flat_size = int(conv3_weight.numel())
            protect_count = max(1, int(round(flat_size * config.protect_top_fraction)))
            generator = torch.Generator()
            if random_seed is not None:
                generator.manual_seed(int(random_seed))
            selected = torch.randperm(flat_size, generator=generator)[:protect_count]
            importance_flat = torch.zeros(flat_size, dtype=torch.float32)
            importance_flat[selected] = 1.0
            return importance_flat.reshape(conv3_weight.shape)

        neuron_freq = torch.tensor(winner_counts, dtype=torch.float32)
        if neuron_freq.numel() < conv3_weight.shape[0]:
            padded = torch.zeros(conv3_weight.shape[0], dtype=torch.float32)
            padded[: neuron_freq.numel()] = neuron_freq
            neuron_freq = padded
        elif neuron_freq.numel() > conv3_weight.shape[0]:
            neuron_freq = neuron_freq[: conv3_weight.shape[0]]

        if config.use_winner_frequency:
            neuron_importance = _max_normalize(neuron_freq)
        else:
            neuron_importance = torch.ones_like(neuron_freq)

        synaptic_strength = conv3_weight.abs()
        if config.use_weight_strength:
            normalized_strength = torch.stack(
                [_max_normalize(synaptic_strength[neuron_idx]) for neuron_idx in range(synaptic_strength.shape[0])],
                dim=0,
            )
        else:
            normalized_strength = torch.ones_like(synaptic_strength)

        importance = neuron_importance.view(-1, 1, 1, 1) * normalized_strength
        importance = _max_normalize(importance)

        protect_fraction = float(config.protect_top_fraction)
        if 0.0 < protect_fraction < 1.0:
            flat = importance.reshape(-1)
            protect_count = max(1, int(round(flat.numel() * protect_fraction)))
            threshold = torch.topk(flat, protect_count).values.min()
            importance = torch.where(importance >= threshold, importance, torch.zeros_like(importance))
            importance = _max_normalize(importance)
        return importance

    @classmethod
    def _importance_to_gate(cls, importance: Tensor, config: SDPMGateConfig) -> Tensor:
        normalized = _max_normalize(importance)
        gamma = float(config.gamma)
        g_min = float(config.g_min)
        return g_min + (1.0 - g_min) * torch.pow(1.0 - normalized, gamma)

    def should_apply(self, stage_name: str) -> bool:
        return self.enabled and stage_name in self.config.apply_stages

    def apply_delta(self, weight_before: Tensor, weight_after: Tensor) -> Tensor:
        if not self.enabled:
            return weight_after
        delta = weight_after - weight_before
        gate = self.gate.to(device=delta.device, dtype=delta.dtype)
        gated = weight_before + gate * delta
        if self.config.track_drift:
            self._update_drift_stats(delta, gate)
        return gated

    def _update_drift_stats(self, delta: Tensor, gate: Tensor) -> None:
        abs_delta = delta.abs()
        if self.importance.numel() == 0:
            return
        importance = self.importance.to(device=abs_delta.device, dtype=abs_delta.dtype)
        protected_mask = importance > 0
        unprotected_mask = ~protected_mask
        protected_drift = float(abs_delta[protected_mask].mean().item()) if protected_mask.any() else 0.0
        unprotected_drift = float(abs_delta[unprotected_mask].mean().item()) if unprotected_mask.any() else 0.0
        total_drift = float(abs_delta.mean().item())
        self.drift_stats = {
            "protected_mean_abs_delta": protected_drift,
            "unprotected_mean_abs_delta": unprotected_drift,
            "overall_mean_abs_delta": total_drift,
            "update_calls": float(self.drift_stats.get("update_calls", 0.0) + 1.0),
        }

    def apply_to_conv3(self, model: nn.Module, weight_before: Tensor, weight_after: Tensor) -> None:
        conv3 = getattr(model, "conv3", None)
        if conv3 is None or not hasattr(conv3, "weight"):
            raise ValueError("SDPMGate.apply_to_conv3 requires model.conv3.weight.")
        gated = self.apply_delta(weight_before, weight_after)
        conv3.weight.copy_(gated)

    @torch.no_grad()
    def gated_reward(self, model: nn.Module) -> None:
        if not hasattr(model, "reward"):
            raise ValueError("Model does not expose reward().")
        conv3 = getattr(model, "conv3", None)
        if conv3 is None:
            model.reward()
            return
        weight_before = conv3.weight.clone()
        model.reward()
        self.apply_to_conv3(model, weight_before, conv3.weight)

    @torch.no_grad()
    def gated_punish(self, model: nn.Module) -> None:
        if not hasattr(model, "punish"):
            raise ValueError("Model does not expose punish().")
        conv3 = getattr(model, "conv3", None)
        if conv3 is None:
            model.punish()
            return
        weight_before = conv3.weight.clone()
        model.punish()
        self.apply_to_conv3(model, weight_before, conv3.weight)

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled or self.gate.numel() == 0:
            return {"enabled": False}

        gate_flat = self.gate.reshape(-1)
        importance_flat = self.importance.reshape(-1)
        protected_fraction = float((importance_flat > 0).float().mean().item())
        return {
            "enabled": True,
            "g_min": self.config.g_min,
            "gamma": self.config.gamma,
            "protect_top_fraction": self.config.protect_top_fraction,
            "random_protection": self.config.random_protection,
            "apply_stages": list(self.config.apply_stages),
            "protected_fraction": protected_fraction,
            "gate_mean": float(gate_flat.mean().item()),
            "gate_min": float(gate_flat.min().item()),
            "gate_max": float(gate_flat.max().item()),
            "importance_mean": float(importance_flat.mean().item()),
            "importance_max": float(importance_flat.max().item()),
            "drift_stats": dict(self.drift_stats),
            "config": asdict(self.config),
        }
