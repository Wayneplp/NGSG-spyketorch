from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .neuron_partition import NeuronPartition
from .occupancy_stats import max_normalize


def _parse_apply_stages(config: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = config.get("apply_stages", ("task2",))
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(stage) for stage in raw)


@dataclass
class NoveltyGateConfig:
    enabled: bool = False
    novelty_threshold: float = 0.15
    use_normalized_occupancy: bool = True
    apply_stages: Tuple[str, ...] = ("task2",)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "NoveltyGateConfig":
        return cls(
            enabled=bool(config.get("enabled", False)),
            novelty_threshold=float(config.get("novelty_threshold", 0.15)),
            use_normalized_occupancy=bool(config.get("use_normalized_occupancy", True)),
            apply_stages=_parse_apply_stages(config),
        )


@dataclass
class NoveltyGate:
    """Score Task 2 samples by old-task occupancy of the natural S3 winner."""

    occupancy: Tensor
    config: NoveltyGateConfig
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @classmethod
    def from_partition(
        cls,
        partition: NeuronPartition,
        config: Mapping[str, Any] | NoveltyGateConfig,
    ) -> "NoveltyGate":
        gate_config = config if isinstance(config, NoveltyGateConfig) else NoveltyGateConfig.from_mapping(config)
        if not gate_config.enabled or not partition.enabled:
            return cls(occupancy=torch.tensor([]), config=gate_config)

        occupancy = partition.combined_score().detach().float().cpu()
        if gate_config.use_normalized_occupancy:
            occupancy = max_normalize(occupancy)
        return cls(occupancy=occupancy, config=gate_config)

    def should_apply(self, stage_name: str) -> bool:
        return self.enabled and stage_name in self.config.apply_stages and self.occupancy.numel() > 0

    def score_winner(self, winner_idx: Optional[int]) -> float:
        if winner_idx is None or winner_idx < 0 or winner_idx >= int(self.occupancy.numel()):
            return 0.0
        return float(self.occupancy[int(winner_idx)].item())

    def is_novel(self, winner_idx: Optional[int]) -> bool:
        return self.score_winner(winner_idx) >= float(self.config.novelty_threshold)

    def observe(self, winner_idx: Optional[int]) -> None:
        score = self.score_winner(winner_idx)
        self.stats["samples"] = float(self.stats.get("samples", 0.0) + 1.0)
        self.stats["score_sum"] = float(self.stats.get("score_sum", 0.0) + score)
        if self.is_novel(winner_idx):
            self.stats["novel_samples"] = float(self.stats.get("novel_samples", 0.0) + 1.0)

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        samples = float(self.stats.get("samples", 0.0))
        novel = float(self.stats.get("novel_samples", 0.0))
        score_sum = float(self.stats.get("score_sum", 0.0))
        return {
            "enabled": True,
            "novelty_threshold": self.config.novelty_threshold,
            "samples": samples,
            "novel_samples": novel,
            "novel_fraction": float(novel / samples) if samples > 0 else 0.0,
            "mean_score": float(score_sum / samples) if samples > 0 else 0.0,
            "occupancy_mean": float(self.occupancy.mean().item()) if self.occupancy.numel() else 0.0,
            "config": asdict(self.config),
        }
