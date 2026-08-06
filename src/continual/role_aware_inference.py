from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .group_confidence import compute_stable_reserve_confidences
from .readout import aggregate_neuron_scores, predict_wta_decision_map
from .task_memory import TaskMemory


@dataclass
class RouteGates:
    stable: float = 0.0
    shared: float = 0.0
    reserve: float = 0.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], defaults: Optional["RouteGates"] = None) -> "RouteGates":
        base = defaults or cls()
        return cls(
            stable=float(config.get("stable", base.stable)),
            shared=float(config.get("shared", base.shared)),
            reserve=float(config.get("reserve", base.reserve)),
        )


@dataclass
class MethodV1RoutingConfig:
    enabled: bool = False
    mode: str = "natural"
    confidence_tau: float = 0.05
    confidence_eps: float = 1e-8
    gate_open_threshold: float = 0.5
    prototype_conflict_tau: float = 0.15
    task1_gates: RouteGates = field(default_factory=lambda: RouteGates(stable=1.0, shared=0.0, reserve=0.0))
    task2_gates: RouteGates = field(default_factory=lambda: RouteGates(stable=0.0, shared=0.6, reserve=1.0))
    uncertain_gates: RouteGates = field(default_factory=lambda: RouteGates(stable=0.7, shared=0.5, reserve=0.3))

    @classmethod
    def from_eval_config(cls, eval_cfg: Mapping[str, Any]) -> "MethodV1RoutingConfig":
        mode = str(eval_cfg.get("routing", "natural")).lower()
        method_cfg = dict(eval_cfg.get("method_v1", {}))
        return cls(
            enabled=mode in {"method_v1", "role_routing", "r2"},
            mode=mode,
            confidence_tau=float(method_cfg.get("confidence_tau", 0.05)),
            confidence_eps=float(method_cfg.get("confidence_eps", 1e-8)),
            gate_open_threshold=float(method_cfg.get("gate_open_threshold", 0.5)),
            prototype_conflict_tau=float(method_cfg.get("prototype_conflict_tau", 0.15)),
            task1_gates=RouteGates.from_mapping(
                method_cfg.get("task1_gates", {}),
                RouteGates(stable=1.0, shared=0.0, reserve=0.0),
            ),
            task2_gates=RouteGates.from_mapping(
                method_cfg.get("task2_gates", {}),
                RouteGates(stable=0.0, shared=0.6, reserve=1.0),
            ),
            uncertain_gates=RouteGates.from_mapping(
                method_cfg.get("uncertain_gates", {}),
                RouteGates(stable=0.7, shared=0.5, reserve=0.3),
            ),
        )


def gates_to_allow_mask(partition, gates: RouteGates, *, open_threshold: float) -> Tensor:
    allow = torch.zeros((partition.num_neurons,), dtype=torch.bool)
    for role_name, gate_value in (
        ("stable", gates.stable),
        ("shared", gates.shared),
        ("reserve", gates.reserve),
    ):
        if float(gate_value) >= float(open_threshold):
            allow |= partition.mask_for_role(role_name)
    return allow


def infer_task_hat_with_decision_map(
    neuron_scores: Tensor,
    partition,
    decision_map: Sequence[int],
    task_memory: Optional[TaskMemory],
    routing_config: MethodV1RoutingConfig,
    *,
    num_classes: int,
) -> Tuple[int, str, Dict[str, Any]]:
    confidences = compute_stable_reserve_confidences(
        neuron_scores,
        partition,
        decision_map,
        num_classes=num_classes,
        eps=routing_config.confidence_eps,
    )
    details: Dict[str, Any] = {"confidences": confidences}
    conf_stable = float(confidences["stable"])
    conf_reserve = float(confidences["reserve"])
    tau = float(routing_config.confidence_tau)
    margin = conf_stable - conf_reserve

    confidence_task: Optional[int] = None
    route_kind = "uncertain"
    if margin > tau:
        confidence_task = 0
        route_kind = "task1"
    elif margin < -tau:
        confidence_task = 1
        route_kind = "task2"

    prototype_task: Optional[int] = None
    sims: Dict[str, float] = {}
    if task_memory is not None and task_memory.enabled:
        sim_tensor = task_memory.similarities(neuron_scores)
        if sim_tensor.numel() >= 2:
            sims = {"task1": float(sim_tensor[0].item()), "task2": float(sim_tensor[1].item())}
            prototype_task = int(sim_tensor.argmax().item())
        details["prototype_similarity"] = sims

    if confidence_task is not None:
        if prototype_task is not None and prototype_task != confidence_task:
            sim_gap = abs(sims.get("task1", 0.0) - sims.get("task2", 0.0))
            if sim_gap >= routing_config.prototype_conflict_tau:
                details["conflict"] = {
                    "confidence_task": confidence_task,
                    "prototype_task": prototype_task,
                    "similarity_gap": sim_gap,
                }
                details["route_reason"] = "confidence_prototype_conflict"
                return confidence_task, "uncertain", details
        details["route_reason"] = "confidence"
        return confidence_task, route_kind, details

    if prototype_task is not None:
        details["route_reason"] = "prototype_fallback"
        route_kind = "task1" if prototype_task == 0 else "task2"
        return prototype_task, route_kind, details

    details["route_reason"] = "uncertain_default_task1"
    return 0, "uncertain", details


def build_route_allow_mask(
    partition,
    routing_config: MethodV1RoutingConfig,
    *,
    route_kind: str,
) -> Tensor:
    if route_kind == "task1":
        gates = routing_config.task1_gates
    elif route_kind == "task2":
        gates = routing_config.task2_gates
    else:
        gates = routing_config.uncertain_gates
    return gates_to_allow_mask(partition, gates, open_threshold=routing_config.gate_open_threshold)


@torch.no_grad()
def predict_with_method_v1_routing(
    model: nn.Module,
    sample: Tensor,
    partition,
    task_memory: Optional[TaskMemory],
    routing_config: MethodV1RoutingConfig,
    *,
    eval_task_index: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    if not hasattr(model, "forward_s3_potentials"):
        raise ValueError("Method v1 routing requires model.forward_s3_potentials().")
    pot = model.forward_s3_potentials(sample)
    num_neurons = int(getattr(getattr(model, "config", None), "s3_neurons", len(getattr(model, "decision_map", []))))
    scores = aggregate_neuron_scores(pot, num_neurons)
    decision_map = getattr(model, "decision_map", [])
    num_classes = int(getattr(getattr(model, "config", None), "num_classes", 10))

    if eval_task_index is not None:
        route_kind = "task1" if int(eval_task_index) == 0 else "task2"
        task_hat = int(eval_task_index)
        details = {"route_reason": "oracle_eval_task", "route_kind": route_kind}
    else:
        task_hat, route_kind, details = infer_task_hat_with_decision_map(
            scores,
            partition,
            decision_map,
            task_memory,
            routing_config,
            num_classes=num_classes,
        )
        details["route_kind"] = route_kind
        details["task_hat"] = int(task_hat)

    allow_mask = build_route_allow_mask(partition, routing_config, route_kind=route_kind)
    prediction = predict_wta_decision_map(scores, decision_map, allow_mask=allow_mask)
    details["allow_neurons"] = int(allow_mask.sum().item())
    return int(prediction), details


def summarize_routing_config(config: MethodV1RoutingConfig) -> Dict[str, Any]:
    return {
        "enabled": config.enabled,
        "mode": config.mode,
        "confidence_tau": config.confidence_tau,
        "gate_open_threshold": config.gate_open_threshold,
        "task1_gates": asdict(config.task1_gates),
        "task2_gates": asdict(config.task2_gates),
        "uncertain_gates": asdict(config.uncertain_gates),
    }
