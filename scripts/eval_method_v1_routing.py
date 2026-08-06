#!/usr/bin/env python3
"""Evaluate METHOD v1 confidence + route-mask inference (R2).

Reports Acc_task, Acc_class, route-kind counts, and mean group confidences
on Task1/MNIST and Task2/EMNIST test splits after continual training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_partition_group_diagnosis import (  # noqa: E402
    extract_partition_payload,
    load_checkpoint,
    load_json,
)
from src.continual.neuron_partition import NeuronPartition  # noqa: E402
from src.continual.task_memory import TaskMemory  # noqa: E402
from src.trainers import TRAINER_REGISTRY  # noqa: E402
from src.utils.data import build_task_bundles  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(run_dir: Path, explicit_config: Optional[Path]) -> Dict[str, Any]:
    if explicit_config is not None:
        return load_yaml(resolve_path(explicit_config))
    resolved = run_dir / "resolved_config.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing resolved_config.json in {run_dir}; pass --config.")
    return load_json(resolved)


def load_task_memory_from_result(
    result: Mapping[str, Any],
    *,
    num_neurons: int,
) -> Optional[TaskMemory]:
    extra = result.get("extra") or {}
    payload = extra.get("task_memory")
    if not isinstance(payload, dict):
        return None
    memory = TaskMemory.from_dict(payload, num_neurons=num_neurons)
    return memory if memory.enabled else None


def run_method_v1_routing_eval(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    checkpoint_stage: str,
    device: str,
) -> Dict[str, Any]:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")

    result = load_json(result_path)
    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()

    _, model, torch_device = load_checkpoint(run_dir, config, device, stage=checkpoint_stage)
    partition = NeuronPartition.from_role_payload(extract_partition_payload(result))
    if partition is None or not partition.enabled:
        raise ValueError("Neuron partition is required for METHOD v1 routing eval.")

    num_neurons = int(getattr(getattr(model, "config", None), "s3_neurons", len(getattr(model, "decision_map", []))))
    task_memory = load_task_memory_from_result(result, num_neurons=num_neurons)

    task_bundles = build_task_bundles(config["data"], config["tasks"])
    if len(task_bundles) < 2:
        raise ValueError("METHOD v1 routing eval requires at least two task bundles.")

    test_task1_loader = trainer.build_eval_loader(task_bundles[0].test_dataset, config)
    test_task2_loader = trainer.build_eval_loader(task_bundles[1].test_dataset, config)

    routing_summary = trainer.summarize_method_v1_routing_eval(
        model,
        test_task1_loader,
        test_task2_loader,
        torch_device,
        config=config,
        partition=partition,
        task_memory=task_memory,
    )
    if routing_summary is None:
        raise ValueError(
            "Routing eval returned None. Ensure eval.routing is method_v1 / role_routing / r2."
        )

    natural_task1 = trainer.evaluate(model, test_task1_loader, torch_device)
    natural_task2 = trainer.evaluate(model, test_task2_loader, torch_device)

    return {
        "run_dir": str(run_dir),
        "checkpoint_stage": checkpoint_stage,
        "method_v1_routing": routing_summary,
        "natural_wta_baseline": {
            "task1_accuracy": float(natural_task1),
            "task2_accuracy": float(natural_task2),
            "avg_accuracy": float((natural_task1 + natural_task2) / 2.0),
        },
        "task_memory_loaded": bool(task_memory is not None and task_memory.enabled),
    }


def write_markdown_summary(payload: Mapping[str, Any], output_path: Path) -> None:
    routing = payload["method_v1_routing"]
    task1 = routing["task1"]
    task2 = routing["task2"]
    natural = payload["natural_wta_baseline"]
    lines = [
        "# METHOD v1 Routing Eval",
        "",
        f"- run_dir: `{payload['run_dir']}`",
        f"- checkpoint: `{payload['checkpoint_stage']}`",
        "",
        "## Routing (R2)",
        "",
        "| split | Acc_class | Acc_task | mean conf_stable | mean conf_reserve |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Task1 / MNIST | {task1['accuracy'] * 100:.2f}% | {task1['acc_task'] * 100:.2f}% "
            f"| {task1['mean_conf_stable']:.4f} | {task1['mean_conf_reserve']:.4f} |"
        ),
        (
            f"| Task2 / EMNIST | {task2['accuracy'] * 100:.2f}% | {task2['acc_task'] * 100:.2f}% "
            f"| {task2['mean_conf_stable']:.4f} | {task2['mean_conf_reserve']:.4f} |"
        ),
        (
            f"| **Avg** | **{routing['avg_acc_class'] * 100:.2f}%** | "
            f"**{routing['avg_acc_task'] * 100:.2f}%** | — | — |"
        ),
        "",
        "## Natural WTA baseline (same checkpoint)",
        "",
        (
            f"- Task1: {natural['task1_accuracy'] * 100:.2f}% · "
            f"Task2: {natural['task2_accuracy'] * 100:.2f}% · "
            f"Avg: {natural['avg_accuracy'] * 100:.2f}%"
        ),
        "",
        "## Route reasons (Task1)",
        "",
        json.dumps(task1.get("route_reason_counts", {}), indent=2, ensure_ascii=False),
        "",
        "## Route reasons (Task2)",
        "",
        json.dumps(task2.get("route_reason_counts", {}), indent=2, ensure_ascii=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate METHOD v1 routing on a completed run.")
    parser.add_argument("--run-dir", required=True, help="Experiment run directory.")
    parser.add_argument("--config", default=None, help="Optional config YAML override.")
    parser.add_argument(
        "--checkpoint-stage",
        choices=["task1", "task2"],
        default="task2",
        help="Which model checkpoint to load.",
    )
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path (default: run_dir/diagnostics/method_v1_routing.json).",
    )
    parser.add_argument("--write-markdown", action="store_true", help="Also write a markdown summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(Path(args.run_dir))
    config = load_config(run_dir, Path(args.config) if args.config else None)
    payload = run_method_v1_routing_eval(
        run_dir=run_dir,
        config=config,
        checkpoint_stage=args.checkpoint_stage,
        device=args.device,
    )

    output_json = (
        resolve_path(Path(args.output_json))
        if args.output_json
        else run_dir / "diagnostics" / "method_v1_routing.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_json}")

    if args.write_markdown:
        md_path = output_json.with_suffix(".md")
        write_markdown_summary(payload, md_path)
        print(f"Wrote {md_path}")

    routing = payload["method_v1_routing"]
    print(
        f"METHOD v1 avg Acc_class={routing['avg_acc_class'] * 100:.2f}% "
        f"Acc_task={routing['avg_acc_task'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
