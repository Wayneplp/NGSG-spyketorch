#!/usr/bin/env python3
"""Evaluate oracle task-id group routing after continual training.

R3 oracle routing uses the known test split identity only for analysis:

Task1 / MNIST  -> stable + shared  (mask reserve)
Task2 / EMNIST -> shared + reserve (mask stable)

It is an upper-bound diagnostic for HTM routing, not a deployable inference
protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_partition_group_diagnosis import (  # noqa: E402
    evaluate_group_diagnosis,
    extract_partition_payload,
    load_checkpoint,
    load_json,
)
from src.continual.neuron_partition import NeuronPartition  # noqa: E402
from src.utils.data import build_task_bundles  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(run_dir: Path, explicit_config: Path | None) -> Dict[str, Any]:
    if explicit_config is not None:
        return load_yaml(resolve_path(explicit_config))

    resolved = run_dir / "resolved_config.json"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing resolved_config.json in {run_dir}; pass --config.")
    return load_json(resolved)


def compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": float(row["accuracy"]),
        "correct": float(row["correct"]),
        "total": float(row["total"]),
        "silent": float(row["silent"]),
        "role_win_fractions": dict(row.get("role_win_fractions", {})),
    }


def evaluate_split(
    *,
    model: Any,
    trainer: Any,
    dataset: Any,
    config: Mapping[str, Any],
    device: Any,
    partition: NeuronPartition,
    decision_map: Any,
) -> Dict[str, Dict[str, Any]]:
    loader = trainer.build_eval_loader(dataset, config)
    rows = evaluate_group_diagnosis(model, trainer, loader, device, partition, decision_map)
    return {key: compact_row(value) for key, value in rows.items() if not key.startswith("_")}


def run_oracle_routing(
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
    trainer, model, torch_device = load_checkpoint(run_dir, config, device, stage=checkpoint_stage)
    partition = NeuronPartition.from_role_payload(extract_partition_payload(result))
    if partition is None or not partition.enabled:
        raise ValueError("Neuron partition is required for oracle group routing.")

    decision_map = getattr(model, "decision_map", None)
    if decision_map is None:
        raise ValueError("Model is missing decision_map.")

    task_bundles = build_task_bundles(config["data"], config["tasks"])
    if len(task_bundles) < 2:
        raise ValueError("Oracle group routing requires at least two task bundles.")

    task1_rows = evaluate_split(
        model=model,
        trainer=trainer,
        dataset=task_bundles[0].test_dataset,
        config=config,
        device=torch_device,
        partition=partition,
        decision_map=decision_map,
    )
    task2_rows = evaluate_split(
        model=model,
        trainer=trainer,
        dataset=task_bundles[1].test_dataset,
        config=config,
        device=torch_device,
        partition=partition,
        decision_map=decision_map,
    )

    task1_natural = float(task1_rows["all_200"]["accuracy"])
    task2_natural = float(task2_rows["all_200"]["accuracy"])
    task1_oracle = float(task1_rows["mask_reserve"]["accuracy"])
    task2_oracle = float(task2_rows["mask_stable"]["accuracy"])
    natural_avg = (task1_natural + task2_natural) / 2.0
    oracle_avg = (task1_oracle + task2_oracle) / 2.0

    metrics = result.get("metrics") or {}
    return {
        "run_name": config.get("run_name") or result.get("run_name"),
        "checkpoint_stage": checkpoint_stage,
        "role_counts": partition.counts_by_role(),
        "task_names": {
            "task1": getattr(task_bundles[0], "name", "task1"),
            "task2": getattr(task_bundles[1], "name", "task2"),
        },
        "embedded_metrics": {
            "task1_after_task1": metrics.get("task1_after_task1"),
            "task1_after_task2": metrics.get("task1_after_task2"),
            "task2_after_task2": metrics.get("task2_after_task2"),
            "avg_acc": metrics.get("avg_acc"),
        },
        "routing_rule": {
            "task1": "stable+shared (mask_reserve)",
            "task2": "shared+reserve (mask_stable)",
        },
        "natural": {
            "task1_all_200": task1_natural,
            "task2_all_200": task2_natural,
            "avg": natural_avg,
        },
        "oracle": {
            "task1_mask_reserve": task1_oracle,
            "task2_mask_stable": task2_oracle,
            "avg": oracle_avg,
            "delta_vs_natural": oracle_avg - natural_avg,
            "delta_vs_natural_pp": (oracle_avg - natural_avg) * 100.0,
        },
        "task1_group_diagnosis": task1_rows,
        "task2_group_diagnosis": task2_rows,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    natural = summary["natural"]
    oracle = summary["oracle"]
    task_names = summary["task_names"]
    lines = [
        f"# R3 oracle group routing: {summary['run_name']}",
        "",
        f"- checkpoint: {summary['checkpoint_stage']}",
        f"- role counts: {summary['role_counts']}",
        f"- Task1: {task_names['task1']} -> stable+shared (mask_reserve)",
        f"- Task2: {task_names['task2']} -> shared+reserve (mask_stable)",
        "",
        "| metric | natural all-200 | oracle routing | delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Task1 | {natural['task1_all_200'] * 100:.2f}% | "
            f"{oracle['task1_mask_reserve'] * 100:.2f}% | "
            f"{(oracle['task1_mask_reserve'] - natural['task1_all_200']) * 100:+.2f} pp |"
        ),
        (
            f"| Task2 | {natural['task2_all_200'] * 100:.2f}% | "
            f"{oracle['task2_mask_stable'] * 100:.2f}% | "
            f"{(oracle['task2_mask_stable'] - natural['task2_all_200']) * 100:+.2f} pp |"
        ),
        (
            f"| Avg | {natural['avg'] * 100:.2f}% | {oracle['avg'] * 100:.2f}% | "
            f"{oracle['delta_vs_natural_pp']:+.2f} pp |"
        ),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment directory with result.json and artifacts.")
    parser.add_argument("--config", type=Path, help="Optional YAML config; defaults to run-dir/resolved_config.json.")
    parser.add_argument(
        "--checkpoint-stage",
        choices=("task1", "task2"),
        default="task2",
        help="Checkpoint to load. R3 should usually use task2.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--write-json", type=Path, help="Optional path to write the full summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    config = load_config(run_dir, args.config)
    summary = run_oracle_routing(
        run_dir=run_dir,
        config=config,
        checkpoint_stage=args.checkpoint_stage,
        device=args.device,
    )

    print(format_summary(summary))

    if args.write_json:
        out = resolve_path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
