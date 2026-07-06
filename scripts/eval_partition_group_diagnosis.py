#!/usr/bin/env python3
"""Task1 group-only / group-masked inference diagnosis (sections 6.1 and 6.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.neuron_partition import NeuronPartition  # noqa: E402
from src.continual.partition_counterfactuals import (  # noqa: E402
    counterfactual_allow_masks,
    frequency_only_partition,
    label_shuffled_fixed_frequency_partition,
    shuffled_history_partition,
)
from src.continual.occupancy_stats import as_2d_label_counts  # noqa: E402
from src.continual.readout import (  # noqa: E402
    PARTITION_GROUP_DIAGNOSES,
    PARTITION_GROUP_LABELS,
    aggregate_neuron_scores,
    extract_winner_label_counts,
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


def save_task1_model_compat(trainer: Any, model: Any, config: Mapping[str, Any]) -> Optional[str]:
    if hasattr(trainer, "_maybe_save_task1_model"):
        return trainer._maybe_save_task1_model(model, config)

    output_cfg = config.get("output", {})
    if not bool(output_cfg.get("save_task1_model", False)):
        return None

    import torch

    run_name = str(config.get("run_name", "unnamed_run"))
    root_dir = Path(str(output_cfg.get("root_dir", "experiments")))
    save_dir = root_dir / run_name / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model_after_task1.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "decision_map": getattr(model, "decision_map", None),
        "run_name": run_name,
        "stage": "task1",
    }
    torch.save(payload, save_path)
    return str(save_path)

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
    save_task1_model_compat(trainer, model, config)
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


def evaluate_with_allow_mask(
    model,
    trainer,
    dataloader,
    device,
    decision_map,
    allow_mask,
    *,
    label: str,
    key: str,
) -> Dict[str, Any]:
    import torch

    from src.trainers.baseline_trainer import move_batch_to_device

    correct = 0
    total = 0
    silent = 0
    for batch in dataloader:
        inputs, targets = move_batch_to_device(batch, device)
        for sample_idx in range(int(targets.shape[0])):
            target = int(targets[sample_idx].item())
            pot, num_neurons = forward_potentials(model, inputs[sample_idx])
            scores = aggregate_neuron_scores(pot, num_neurons)
            prediction = predict_wta_decision_map(scores, decision_map, allow_mask=allow_mask)
            total += 1
            if prediction < 0:
                silent += 1
            elif prediction == target:
                correct += 1
    denom = max(total, 1)
    return {
        "label": label,
        "accuracy": float(correct / denom),
        "correct": float(correct),
        "total": float(total),
        "silent": float(silent),
        "_key": key,
    }


def summarize_random_trials(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import torch

    accs = [float(row["accuracy"]) for row in trials]
    if not accs:
        return {"n": 0, "mean_accuracy": float("nan"), "std_accuracy": float("nan"), "trials": []}
    tensor = torch.tensor(accs)
    return {
        "n": len(accs),
        "mean_accuracy": float(tensor.mean().item()),
        "std_accuracy": float(tensor.std(unbiased=False).item()) if len(accs) > 1 else 0.0,
        "min_accuracy": float(tensor.min().item()),
        "max_accuracy": float(tensor.max().item()),
        "trials": list(trials),
    }


def _fallback_label_counts(partition: NeuronPartition, winner_label_counts, num_classes: int):
    import torch

    label_counts = as_2d_label_counts(winner_label_counts, partition.num_neurons, num_classes)
    if float(label_counts.sum().item()) > 0.0:
        return label_counts.float()

    label_counts = torch.zeros(partition.num_neurons, num_classes, dtype=torch.float32)
    for neuron_idx in range(partition.num_neurons):
        wins = float(partition.f_i[neuron_idx].item())
        dom = int(partition.dominant_labels[neuron_idx].item())
        if wins > 0 and 0 <= dom < num_classes:
            label_counts[neuron_idx, dom] = wins
    return label_counts


def mask_class_coverage(
    partition: NeuronPartition,
    winner_label_counts,
    decision_map,
    allow_mask,
    *,
    num_classes: int,
) -> Dict[str, Any]:
    import torch

    label_counts = _fallback_label_counts(partition, winner_label_counts, num_classes)
    mask = allow_mask.detach().cpu().bool()
    if mask.numel() != partition.num_neurons:
        raise ValueError("allow_mask length must match partition.num_neurons")

    decision = torch.full((partition.num_neurons,), -1, dtype=torch.long)
    for idx, label in enumerate(decision_map):
        if idx < partition.num_neurons:
            decision[idx] = int(label)
    dominant = partition.dominant_labels.detach().cpu().long()
    active = partition.f_i.detach().cpu() > 0

    per_class = []
    stable_win_fracs = []
    for cls in range(num_classes):
        class_counts = label_counts[:, cls]
        total_wins = float(class_counts.sum().item())
        mask_wins = float(class_counts[mask].sum().item())
        win_fraction = float(mask_wins / total_wins) if total_wins > 0 else 0.0
        stable_win_fracs.append(win_fraction)
        if total_wins > 0:
            top_idx = int(class_counts.argmax().item())
            top_role = partition.role_name(top_idx)
        else:
            top_idx = -1
            top_role = "none"
        per_class.append(
            {
                "class": int(cls),
                "total_winner_events": total_wins,
                "covered_winner_events": mask_wins,
                "covered_winner_fraction": win_fraction,
                "covered_winner_neurons": int(((class_counts > 0) & mask).sum().item()),
                "active_winner_neurons": int((class_counts > 0).sum().item()),
                "covered_static_label_neurons": int(((decision == cls) & mask).sum().item()),
                "covered_dominant_label_neurons": int(((dominant == cls) & active & mask).sum().item()),
                "top_winner_neuron": top_idx,
                "top_winner_role": top_role,
                "top_winner_covered": bool(top_idx >= 0 and bool(mask[top_idx].item())),
            }
        )

    return {
        "num_neurons": int(mask.sum().item()),
        "classes_with_winner_coverage": int(sum(1 for row in per_class if row["covered_winner_neurons"] > 0)),
        "classes_with_static_label_coverage": int(sum(1 for row in per_class if row["covered_static_label_neurons"] > 0)),
        "classes_with_dominant_label_coverage": int(sum(1 for row in per_class if row["covered_dominant_label_neurons"] > 0)),
        "classes_with_top_winner_covered": int(sum(1 for row in per_class if row["top_winner_covered"])),
        "min_winner_event_fraction": float(min(stable_win_fracs) if stable_win_fracs else 0.0),
        "mean_winner_event_fraction": float(sum(stable_win_fracs) / max(len(stable_win_fracs), 1)),
        "per_class": per_class,
    }


def summarize_mask_class_coverages(
    partition: NeuronPartition,
    winner_label_counts,
    decision_map,
    masks: Sequence[Tuple[int, Any]],
    *,
    num_classes: int,
) -> Dict[str, Any]:
    import torch

    rows = []
    for seed, mask in masks:
        coverage = mask_class_coverage(
            partition,
            winner_label_counts,
            decision_map,
            mask,
            num_classes=num_classes,
        )
        rows.append(
            {
                "seed": int(seed),
                "num_neurons": int(coverage["num_neurons"]),
                "classes_with_winner_coverage": int(coverage["classes_with_winner_coverage"]),
                "classes_with_static_label_coverage": int(coverage["classes_with_static_label_coverage"]),
                "classes_with_dominant_label_coverage": int(coverage["classes_with_dominant_label_coverage"]),
                "classes_with_top_winner_covered": int(coverage["classes_with_top_winner_covered"]),
                "min_winner_event_fraction": float(coverage["min_winner_event_fraction"]),
                "mean_winner_event_fraction": float(coverage["mean_winner_event_fraction"]),
            }
        )
    if not rows:
        return {"n": 0, "trials": []}

    def summarize_field(field: str) -> Dict[str, float]:
        tensor = torch.tensor([float(row[field]) for row in rows], dtype=torch.float32)
        return {
            "mean": float(tensor.mean().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }

    return {
        "n": len(rows),
        "classes_with_winner_coverage": summarize_field("classes_with_winner_coverage"),
        "classes_with_static_label_coverage": summarize_field("classes_with_static_label_coverage"),
        "classes_with_dominant_label_coverage": summarize_field("classes_with_dominant_label_coverage"),
        "classes_with_top_winner_covered": summarize_field("classes_with_top_winner_covered"),
        "min_winner_event_fraction": summarize_field("min_winner_event_fraction"),
        "mean_winner_event_fraction": summarize_field("mean_winner_event_fraction"),
        "trials": rows,
    }


def evaluate_per_class_with_allow_mask(
    model,
    trainer,
    dataloader,
    device,
    decision_map,
    allow_mask,
    *,
    num_classes: int,
) -> Dict[str, Any]:
    from src.trainers.baseline_trainer import move_batch_to_device

    correct = [0 for _ in range(num_classes)]
    total = [0 for _ in range(num_classes)]
    silent = [0 for _ in range(num_classes)]
    for batch in dataloader:
        inputs, targets = move_batch_to_device(batch, device)
        for sample_idx in range(int(targets.shape[0])):
            target = int(targets[sample_idx].item())
            if target < 0 or target >= num_classes:
                continue
            pot, num_neurons = forward_potentials(model, inputs[sample_idx])
            scores = aggregate_neuron_scores(pot, num_neurons)
            prediction = predict_wta_decision_map(scores, decision_map, allow_mask=allow_mask)
            total[target] += 1
            if prediction < 0:
                silent[target] += 1
            elif prediction == target:
                correct[target] += 1

    per_class = []
    for cls in range(num_classes):
        denom = max(total[cls], 1)
        per_class.append(
            {
                "class": int(cls),
                "accuracy": float(correct[cls] / denom),
                "correct": int(correct[cls]),
                "total": int(total[cls]),
                "silent": int(silent[cls]),
            }
        )
    return {"per_class": per_class}

def run_counterfactuals(
    *,
    model,
    trainer,
    test_loader,
    device,
    partition: NeuronPartition,
    decision_map,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    counterfactual_seeds: Sequence[int],
) -> Dict[str, Any]:
    import torch

    num_classes = int(config.get("data", {}).get("num_classes", 10))
    winner_label_counts = extract_winner_label_counts(result, "task1")

    payload: Dict[str, Any] = {}

    random_masks = counterfactual_allow_masks(partition, seeds=counterfactual_seeds)
    stable_mask = partition.mask_for_role("stable")
    payload["per_class_stable_coverage"] = mask_class_coverage(
        partition,
        winner_label_counts,
        decision_map,
        stable_mask,
        num_classes=num_classes,
    )
    payload["per_class_stable_coverage"]["per_class_accuracy"] = {
        "all_200": evaluate_per_class_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            None,
            num_classes=num_classes,
        )["per_class"],
        "stable_only": evaluate_per_class_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            stable_mask,
            num_classes=num_classes,
        )["per_class"],
        "mask_stable": evaluate_per_class_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            ~stable_mask,
            num_classes=num_classes,
        )["per_class"],
    }    random_stable_trials = []
    for seed, pick in random_masks["random_stable_only"]:
        row = evaluate_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            pick,
            label=f"random {int(pick.sum().item())}-neuron only (seed {seed})",
            key=f"random_stable_only_seed{seed}",
        )
        row["seed"] = int(seed)
        random_stable_trials.append(row)
    payload["random_stable_only"] = summarize_random_trials(random_stable_trials)
    payload["random_stable_only"]["reference"] = "stable_only"

    random_mask_trials = []
    for seed, pick in random_masks["random_mask_stable"]:
        row = evaluate_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            ~pick,
            label=f"random mask {int(pick.sum().item())} neurons (seed {seed})",
            key=f"random_mask_stable_seed{seed}",
        )
        row["seed"] = int(seed)
        random_mask_trials.append(row)
    payload["random_mask_stable"] = summarize_random_trials(random_mask_trials)
    payload["random_mask_stable"]["reference"] = "mask_stable"

    matched_active_stable_trials = []
    for seed, pick in random_masks["matched_active_random_stable_only"]:
        row = evaluate_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            pick,
            label=f"matched-active random {int(pick.sum().item())}-neuron only (seed {seed})",
            key=f"matched_active_random_stable_only_seed{seed}",
        )
        row["seed"] = int(seed)
        matched_active_stable_trials.append(row)
    payload["matched_active_random_stable_only"] = summarize_random_trials(matched_active_stable_trials)
    payload["matched_active_random_stable_only"]["reference"] = "stable_only"

    matched_active_mask_trials = []
    for seed, pick in random_masks["matched_active_random_mask_stable"]:
        row = evaluate_with_allow_mask(
            model,
            trainer,
            test_loader,
            device,
            decision_map,
            ~pick,
            label=f"matched-active random mask {int(pick.sum().item())} neurons (seed {seed})",
            key=f"matched_active_random_mask_stable_seed{seed}",
        )
        row["seed"] = int(seed)
        matched_active_mask_trials.append(row)
    payload["matched_active_random_mask_stable"] = summarize_random_trials(matched_active_mask_trials)
    payload["matched_active_random_mask_stable"]["reference"] = "mask_stable"

    payload["random_class_coverage"] = {
        "random_stable_only": summarize_mask_class_coverages(
            partition,
            winner_label_counts,
            decision_map,
            random_masks["random_stable_only"],
            num_classes=num_classes,
        ),
        "matched_active_random_stable_only": summarize_mask_class_coverages(
            partition,
            winner_label_counts,
            decision_map,
            random_masks["matched_active_random_stable_only"],
            num_classes=num_classes,
        ),
    }
    shuffled_rows = {}
    for seed in counterfactual_seeds[:3]:
        shuffled = shuffled_history_partition(
            partition,
            seed=int(seed),
            winner_label_counts=winner_label_counts,
            num_classes=num_classes,
        )
        diag = evaluate_group_diagnosis(model, trainer, test_loader, device, shuffled, decision_map)
        shuffled_rows[f"seed_{seed}"] = {
            "role_counts": shuffled.counts_by_role(),
            "stable_only": diag["stable_only"]["accuracy"],
            "mask_stable": diag["mask_stable"]["accuracy"],
            "all_200": diag["all_200"]["accuracy"],
        }
    payload["shuffled_history_partition"] = shuffled_rows

    freq_partition = frequency_only_partition(partition)
    freq_diag = evaluate_group_diagnosis(model, trainer, test_loader, device, freq_partition, decision_map)
    payload["frequency_only_partition"] = {
        "role_counts": freq_partition.counts_by_role(),
        "thresholds": freq_partition.thresholds,
        "diagnosis": {key: freq_diag[key] for key in freq_diag if not key.startswith("_")},
    }
    payload["frequency_only_partition"]["stable_only"] = float(freq_diag["stable_only"]["accuracy"])
    payload["frequency_only_partition"]["mask_stable"] = float(freq_diag["mask_stable"]["accuracy"])

    return payload


def format_counterfactual_section(counterfactuals: Mapping[str, Any], *, baseline_rows: Mapping[str, Any]) -> str:
    stable_only = float(baseline_rows["stable_only"]["accuracy"])
    mask_stable = float(baseline_rows["mask_stable"]["accuracy"])
    all_acc = float(baseline_rows["all_200"]["accuracy"])

    lines = [
        "## Counterfactual controls (WTA partition falsification)",
        "",
        f"- reference all-200: {all_acc * 100:.2f}%",
        f"- reference stable-only: {stable_only * 100:.2f}%",
        f"- reference mask-stable: {mask_stable * 100:.2f}%",
        "",
        "| control | reference | mean acc | std | delta vs reference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, ref_key, ref_acc in (
        ("random_stable_only", "stable_only", stable_only),
        ("matched_active_random_stable_only", "stable_only", stable_only),
        ("random_mask_stable", "mask_stable", mask_stable),
        ("matched_active_random_mask_stable", "mask_stable", mask_stable),
    ):
        block = counterfactuals.get(key, {})
        mean_acc = float(block.get("mean_accuracy", float("nan")))
        std_acc = float(block.get("std_accuracy", float("nan")))
        lines.append(
            f"| {key} (n={block.get('n', 0)}) | {ref_key} | {mean_acc * 100:.2f}% | "
            f"{std_acc * 100:.2f} pp | {(mean_acc - ref_acc) * 100:+.2f} pp |"
        )

    shuffled = counterfactuals.get("shuffled_history_partition", {})
    if shuffled:
        lines.extend(["", "### shuffled winner-history partition", ""])
        lines.append("| seed | stable-only | mask-stable | role counts |")
        lines.append("| --- | ---: | ---: | --- |")
        for seed_key, row in shuffled.items():
            counts = row.get("role_counts", {})
            lines.append(
                f"| {seed_key} | {float(row['stable_only']) * 100:.2f}% | "
                f"{float(row['mask_stable']) * 100:.2f}% | {counts} |"
            )

    freq = counterfactuals.get("frequency_only_partition", {})
    if freq:
        lines.extend(["", "### frequency-only partition", ""])
        lines.append(
            f"- stable-only: {float(freq.get('stable_only', float('nan'))) * 100:.2f}% "
            f"(delta vs WTA stable-only {(float(freq.get('stable_only', 0)) - stable_only) * 100:+.2f} pp)"
        )
        lines.append(
            f"- mask-stable: {float(freq.get('mask_stable', float('nan'))) * 100:.2f}% "
            f"(delta vs WTA mask-stable {(float(freq.get('mask_stable', 0)) - mask_stable) * 100:+.2f} pp)"
        )
        lines.append(f"- role counts: {freq.get('role_counts', {})}")
    return "\n".join(lines)


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
    counterfactuals: bool = False,
    counterfactual_seeds: Optional[Sequence[int]] = None,
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
    if counterfactuals:
        seeds = list(counterfactual_seeds or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        summary["counterfactuals"] = run_counterfactuals(
            model=model,
            trainer=trainer,
            test_loader=test_loader,
            device=device,
            partition=partition,
            decision_map=decision_map,
            result=result,
            config=config,
            counterfactual_seeds=seeds,
        )
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
    parser.add_argument(
        "--counterfactuals",
        action="store_true",
        help="Run random/shuffled/frequency-only partition falsification controls.",
    )
    parser.add_argument(
        "--counterfactual-seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated RNG seeds for random same-size controls.",
    )
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

    cf_seeds = [int(x.strip()) for x in str(args.counterfactual_seeds).split(",") if x.strip()]
    summary = run_diagnosis(
        config=config,
        run_dir=run_dir,
        device=args.device,
        train_task1=args.train_task1,
        checkpoint_stage=checkpoint_stage,
        counterfactuals=args.counterfactuals,
        counterfactual_seeds=cf_seeds,
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

    if summary.get("counterfactuals"):
        print()
        print(format_counterfactual_section(summary["counterfactuals"], baseline_rows=rows))

    if args.write_json:
        out = args.write_json if args.write_json.is_absolute() else PROJECT_ROOT / args.write_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




