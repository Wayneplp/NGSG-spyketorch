#!/usr/bin/env python3
"""Task1 group-only / group-masked inference diagnosis (sections 6.1 and 6.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.neuron_partition import NeuronPartition  # noqa: E402
from src.continual.readout import (  # noqa: E402
    PARTITION_GROUP_DIAGNOSES,
    PARTITION_GROUP_LABELS,
    aggregate_neuron_scores,
    predict_wta_decision_map,
    role_allow_mask,
)
from src.continual.reserve_activation import _resolve_num_s3_neurons  # noqa: E402
from src.trainers import TRAINER_REGISTRY  # noqa: E402
from src.utils.data import build_task_bundles  # noqa: E402
from src.utils.runtime import set_seed  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def forward_potentials(model, sample):
    import torch

    with torch.no_grad():
        if hasattr(model, "forward_s3_potentials"):
            pot = model.forward_s3_potentials(sample)
        else:
            was_training = model.training
            model.train()
            try:
                _ = model(sample, 3)
            finally:
                model.train(was_training)
            pot = getattr(model, "ctx", {}).get("potentials")
    if pot is None:
        raise RuntimeError("Model forward did not expose S3 potentials.")
    num_neurons = _resolve_num_s3_neurons(model, pot)
    return pot, num_neurons


def extract_partition_payload(result: Mapping[str, Any]) -> Dict[str, Any]:
    trainer = (result.get("extra") or {}).get("trainer") or {}
    task1 = trainer.get("task1_training") or {}
    partition = task1.get("neuron_partition")
    if isinstance(partition, dict) and partition.get("roles"):
        return partition
    partition = trainer.get("neuron_partition")
    if isinstance(partition, dict) and partition.get("roles"):
        return partition
    raise ValueError("result.json is missing task1 neuron_partition.roles.")


def load_checkpoint(
    run_dir: Path,
    config: Mapping[str, Any],
    device: str,
    *,
    stage: str,
):
    import torch

    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()
    torch_device = trainer.resolve_device({"train": {"device": device}})
    model = trainer.build_model(config).to(torch_device)

    model_name = "model_after_task1.pt" if stage == "task1" else "model_after_task2.pt"
    model_path = run_dir / "artifacts" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    payload = torch.load(model_path, map_location=torch_device)
    model.load_state_dict(payload["model_state_dict"])
    if payload.get("decision_map") is not None:
        model.decision_map = payload["decision_map"]
    model.eval()
    return trainer, model, torch_device


def train_task1_checkpoint(config: Dict[str, Any], device: str) -> Tuple[Any, Any, Any, Dict[str, Any], float]:
    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()
    trainer._ensure_runtime_dependencies()
    set_seed(int(config.get("seed", 0)))

    import torch

    torch_device = trainer.resolve_device({"train": {"device": device}})
    task_bundles = build_task_bundles(config["data"], config["tasks"])
    task1 = task_bundles[0]
    model = trainer.build_model(config).to(torch_device)
    if getattr(model, "paper_source_compatible", False):
        task1, _ = trainer.prepare_paper_source_cache(task1, task_bundles[1], config, model)
    rstdp = None if getattr(model, "paper_source_compatible", False) else trainer.build_output_rstdp(model, config).to(torch_device)

    train_loader = trainer.build_train_loader(task1.train_dataset, config)
    test_loader = trainer.build_eval_loader(task1.test_dataset, config)

    task1_stats = trainer.train_single_task(
        model=model,
        dataloader=train_loader,
        config=config,
        rstdp=rstdp,
        device=torch_device,
        stage_name="task1",
    )
    partition = trainer.fit_neuron_partition_after_task1(model, config, task1_stats)
    if partition is not None and partition.enabled:
        task1_stats["neuron_partition"] = partition.to_dict(include_arrays=True)
    trainer._maybe_save_task1_model(model, config)
    embedded_acc = float(trainer.evaluate(model, test_loader, torch_device))
    return trainer, model, partition, task1_stats, embedded_acc, torch_device


def evaluate_group_diagnosis(
    model,
    trainer,
    dataloader,
    device,
    partition: NeuronPartition,
    decision_map,
) -> Dict[str, Dict[str, float]]:
    import torch

    from src.trainers.baseline_trainer import move_batch_to_device

    results: Dict[str, Dict[str, float]] = {}
    for key, mode, role in PARTITION_GROUP_DIAGNOSES:
        correct = 0
        total = 0
        silent = 0
        role_wins = {"stable": 0, "shared": 0, "reserve": 0, "dead": 0}

        for batch in dataloader:
            inputs, targets = move_batch_to_device(batch, device)
            for sample_idx in range(int(targets.shape[0])):
                target = int(targets[sample_idx].item())
                pot, num_neurons = forward_potentials(model, inputs[sample_idx])
                scores = aggregate_neuron_scores(pot, num_neurons)
                if mode is None:
                    allow_mask = None
                else:
                    allow_mask = role_allow_mask(partition, mode=mode, role=role)
                prediction = predict_wta_decision_map(
                    scores,
                    decision_map,
                    allow_mask=allow_mask,
                )
                total += 1
                if prediction < 0:
                    silent += 1
                elif prediction == target:
                    correct += 1

                winner_scores = scores if allow_mask is None else scores.masked_fill(~allow_mask, float("-inf"))
                if torch.isfinite(winner_scores).any():
                    winner_idx = int(winner_scores.argmax().item())
                    role_wins[partition.role_name(winner_idx)] += 1

        denom = max(total, 1)
        win_denom = max(sum(role_wins.values()), 1)
        results[key] = {
            "label": PARTITION_GROUP_LABELS[key],
            "accuracy": float(correct / denom),
            "correct": float(correct),
            "total": float(total),
            "silent": float(silent),
            "role_win_fractions": {role: float(count / win_denom) for role, count in role_wins.items()},
        }
    return results


def format_section(title: str, rows: Mapping[str, Mapping[str, Any]], *, baseline_key: str) -> str:
    baseline = float(rows[baseline_key]["accuracy"])
    lines = [
        f"## {title}",
        "",
        "| condition | neurons used | accuracy | delta vs all-200 | silent | winner stable% | winner shared% | winner reserve% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    role_counts = rows.get("_role_counts", {})
    for key, payload in rows.items():
        if key.startswith("_"):
            continue
        acc = float(payload["accuracy"])
        fr = payload.get("role_win_fractions", {})
        neuron_note = {
            "all_200": "200",
            "stable_only": str(role_counts.get("stable", "?")),
            "shared_only": str(role_counts.get("shared", "?")),
            "reserve_only": str(role_counts.get("reserve", "?")),
            "mask_stable": f"200-{role_counts.get('stable', '?')}",
            "mask_shared": f"200-{role_counts.get('shared', '?')}",
            "mask_reserve": f"200-{role_counts.get('reserve', '?')}",
        }.get(key, "?")
        lines.append(
            f"| {payload['label']} | {neuron_note} | {acc * 100:.2f}% | {(acc - baseline) * 100:+.2f} pp | "
            f"{int(payload['silent'])} | {fr.get('stable', 0.0) * 100:.1f}% | "
            f"{fr.get('shared', 0.0) * 100:.1f}% | {fr.get('reserve', 0.0) * 100:.1f}% |"
        )
    return "\n".join(lines)


def run_diagnosis(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    device: str,
    train_task1: bool,
    checkpoint_stage: str,
) -> Dict[str, Any]:
    import torch

    if train_task1:
        trainer, model, partition, task1_stats, embedded_acc, device = train_task1_checkpoint(dict(config), device)
        role_counts = partition.counts_by_role() if partition is not None else {}
        result_path = run_dir / "result.json"
        result = {
            "run_name": config.get("run_name"),
            "extra": {
                "trainer": {
                    "task1_training": task1_stats,
                    "neuron_partition": partition.summarize() if partition is not None else {},
                }
            },
        }
        if result_path.exists():
            result = load_json(result_path)
    else:
        result_path = run_dir / "result.json"
        config_path = run_dir / "resolved_config.json"
        if config_path.exists():
            config = load_json(config_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result.json: {result_path}")
        result = load_json(result_path)
        trainer, model, torch_device = load_checkpoint(run_dir, config, device, stage=checkpoint_stage)
        partition = NeuronPartition.from_role_payload(extract_partition_payload(result))
        embedded_acc = float((result.get("metrics") or {}).get("task1_after_task1", float("nan")))
        role_counts = partition.counts_by_role()

    if partition is None or not partition.enabled:
        raise ValueError("Neuron partition is required for group diagnosis.")

    task_bundles = build_task_bundles(config["data"], config["tasks"])
    test_loader = trainer.build_eval_loader(task_bundles[0].test_dataset, config)
    decision_map = getattr(model, "decision_map", None)
    if decision_map is None:
        raise ValueError("Model is missing decision_map.")

    rows = evaluate_group_diagnosis(model, trainer, test_loader, device, partition, decision_map)
    rows["_role_counts"] = role_counts

    summary = {
        "run_name": config.get("run_name") or result.get("run_name"),
        "checkpoint_stage": checkpoint_stage,
        "role_counts": role_counts,
        "embedded_task1_accuracy": embedded_acc,
        "task1_test_after_task1": {key: rows[key] for key in rows if not key.startswith("_")},
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML config (used with --train-task1).")
    parser.add_argument("--run-dir", type=Path, help="Experiment directory with checkpoint + partition.")
    parser.add_argument("--train-task1", action="store_true", help="Train Task1, save model_after_task1.pt, then diagnose.")
    parser.add_argument(
        "--checkpoint-stage",
        choices=("task1", "task2"),
        default="task1",
        help="Which saved checkpoint to load when not training.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--write-json", type=Path)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    if args.train_task1:
        if args.config is None:
            print("error: --train-task1 requires --config", file=sys.stderr)
            return 2
        config = load_yaml(args.config if args.config.is_absolute() else PROJECT_ROOT / args.config)
        run_name = str(config.get("run_name", "partition_group_diagnosis"))
        run_dir = PROJECT_ROOT / str(config.get("output", {}).get("root_dir", "experiments")) / run_name
        checkpoint_stage = "task1"
    else:
        if args.run_dir is None:
            print("error: provide --run-dir or use --train-task1 --config", file=sys.stderr)
            return 2
        run_dir = resolve_run_dir(args.run_dir)
        config_path = run_dir / "resolved_config.json"
        if args.config is not None:
            config = load_yaml(args.config if args.config.is_absolute() else PROJECT_ROOT / args.config)
        elif config_path.exists():
            config = load_json(config_path)
        else:
            print(f"error: missing resolved_config.json in {run_dir}", file=sys.stderr)
            return 2
        checkpoint_stage = args.checkpoint_stage

    summary = run_diagnosis(
        config=config,
        run_dir=run_dir,
        device=args.device,
        train_task1=args.train_task1,
        checkpoint_stage=checkpoint_stage,
    )

    rows = dict(summary["task1_test_after_task1"])
    rows["_role_counts"] = summary["role_counts"]
    print(f"# Partition group diagnosis: {summary['run_name']}")
    print(f"- checkpoint: {checkpoint_stage}")
    print(f"- role counts: {summary['role_counts']}")
    print(f"- embedded trainer.evaluate Task1 acc: {summary['embedded_task1_accuracy']:.4f}")
    print()
    print(format_section("6.1 group-only inference", rows, baseline_key="all_200"))
    print()
    print(format_section("6.2 group-masked inference", rows, baseline_key="all_200"))

    stable_only = float(rows["stable_only"]["accuracy"])
    shared_only = float(rows["shared_only"]["accuracy"])
    reserve_only = float(rows["reserve_only"]["accuracy"])
    all_acc = float(rows["all_200"]["accuracy"])
    mask_reserve = float(rows["mask_reserve"]["accuracy"])
    print()
    print("## Interpretation")
    if shared_only < 0.15 and reserve_only < 0.15 and stable_only > all_acc * 0.8:
        print("- stable-only carries most Task1 accuracy; shared/reserve-only are near chance.")
    else:
        print("- shared-only and/or reserve-only are not negligible on Task1; they still encode Task1 signal.")
    if mask_reserve >= all_acc - 0.02:
        print("- mask-reserve barely changes Task1: reserve pool is not carrying old-task readout.")
    else:
        print("- mask-reserve drops Task1 materially: reserve neurons participate in Task1 inference.")

    if args.write_json:
        out = args.write_json if args.write_json.is_absolute() else PROJECT_ROOT / args.write_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
