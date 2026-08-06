from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .readout import aggregate_neuron_scores


@dataclass
class TaskMemoryConfig:
    enabled: bool = True
    eta: float = 0.1
    eps: float = 1e-8

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TaskMemoryConfig":
        return cls(
            enabled=bool(config.get("enabled", True)),
            eta=float(config.get("eta", 0.1)),
            eps=float(config.get("eps", 1e-8)),
        )


@dataclass
class TaskMemory:
    """Hippocampus-inspired task memory: running-average S3 response prototypes m_k."""

    prototypes: Tensor
    update_counts: Tensor
    config: TaskMemoryConfig
    num_neurons: int

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.prototypes.numel() > 0

    @classmethod
    def create(cls, num_neurons: int, config: TaskMemoryConfig, *, num_tasks: int = 2) -> "TaskMemory":
        return cls(
            prototypes=torch.zeros((num_tasks, num_neurons), dtype=torch.float32),
            update_counts=torch.zeros((num_tasks,), dtype=torch.float32),
            config=config,
            num_neurons=int(num_neurons),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, num_neurons: int, num_tasks: int = 2) -> "TaskMemory":
        memory_config = TaskMemoryConfig.from_mapping(config)
        if not memory_config.enabled:
            return cls(
                prototypes=torch.tensor([]),
                update_counts=torch.tensor([]),
                config=memory_config,
                num_neurons=int(num_neurons),
            )
        return cls.create(num_neurons, memory_config, num_tasks=num_tasks)

    def _response_from_potentials(self, pot: Tensor) -> Tensor:
        return aggregate_neuron_scores(pot, self.num_neurons).float()

    @torch.no_grad()
    def update_from_potentials(self, pot: Tensor, task_idx: int) -> None:
        if not self.enabled:
            return
        if task_idx < 0 or task_idx >= int(self.prototypes.shape[0]):
            raise ValueError(f"task_idx={task_idx} out of range for {self.prototypes.shape[0]} prototypes.")
        response = self._response_from_potentials(pot)
        eta = float(self.config.eta)
        self.prototypes[task_idx] = (1.0 - eta) * self.prototypes[task_idx] + eta * response.to(self.prototypes.device)
        self.update_counts[task_idx] += 1.0

    @torch.no_grad()
    def update_from_model_sample(self, model: nn.Module, sample: Tensor, task_idx: int) -> None:
        if not self.enabled:
            return
        if not hasattr(model, "forward_s3_potentials"):
            raise ValueError("TaskMemory requires model.forward_s3_potentials().")
        was_training = model.training
        model.eval()
        pot = model.forward_s3_potentials(sample)
        if was_training:
            model.train()
        self.update_from_potentials(pot, task_idx)

    def cosine_similarity(self, response: Tensor, task_idx: int) -> float:
        if not self.enabled:
            return 0.0
        prototype = self.prototypes[task_idx].to(device=response.device, dtype=response.dtype)
        norm_r = torch.linalg.norm(response)
        norm_p = torch.linalg.norm(prototype)
        if float(norm_r.item()) <= self.config.eps or float(norm_p.item()) <= self.config.eps:
            return 0.0
        sim = torch.dot(response, prototype) / (norm_r * norm_p + self.config.eps)
        return float(sim.item())

    def similarities(self, response: Tensor) -> Tensor:
        if not self.enabled:
            return torch.zeros((0,), dtype=torch.float32)
        sims = []
        for task_idx in range(int(self.prototypes.shape[0])):
            sims.append(self.cosine_similarity(response, task_idx))
        return torch.tensor(sims, dtype=torch.float32)

    def infer_task_index(self, response: Tensor) -> int:
        sims = self.similarities(response)
        if sims.numel() == 0:
            return 0
        return int(sims.argmax().item())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config": asdict(self.config),
            "num_neurons": self.num_neurons,
            "update_counts": [float(x) for x in self.update_counts.tolist()],
            "prototypes": self.prototypes.detach().cpu().tolist() if self.prototypes.numel() else [],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, num_neurons: int) -> "TaskMemory":
        config = TaskMemoryConfig.from_mapping(payload.get("config", payload))
        prototypes = payload.get("prototypes", [])
        if not config.enabled or not prototypes:
            return cls.from_mapping({"enabled": False}, num_neurons=num_neurons)
        proto_tensor = torch.tensor(prototypes, dtype=torch.float32)
        counts = payload.get("update_counts", [0.0] * int(proto_tensor.shape[0]))
        return cls(
            prototypes=proto_tensor,
            update_counts=torch.tensor(counts, dtype=torch.float32),
            config=config,
            num_neurons=num_neurons,
        )

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        sim_01 = 0.0
        if self.prototypes.shape[0] >= 2:
            sim_01 = self.cosine_similarity(self.prototypes[0], 1)
        return {
            "enabled": True,
            "eta": self.config.eta,
            "num_tasks": int(self.prototypes.shape[0]),
            "update_counts": [float(x) for x in self.update_counts.tolist()],
            "prototype_separation_cos": float(sim_01),
        }
