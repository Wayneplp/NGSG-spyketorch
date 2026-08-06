#!/usr/bin/env python3
"""Evaluate Single Role-path Selector on a role_train checkpoint (no K8 stack)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_method_v1_routing import load_config  # noqa: E402
from scripts.eval_partition_group_diagnosis import (  # noqa: E402
    extract_partition_payload,
    forward_potentials,
    load_checkpoint,
    load_json,
)
from src.analysis.role_path_selector import (  # noqa: E402
    ROLE_PATH_FEATURE_NAMES,
    RolePathSelectorMLP,
    apply_role_path_policy,
    choose_probability_threshold,
    evaluate_fixed_path,
    extract_role_path_features,
    role_path_label,
    stratified_disagree_split,
)
from src.continual.neuron_partition import NeuronPartition  # noqa: E402
from src.continual.readout import aggregate_neuron_scores, predict_wta_decision_map  # noqa: E402
from src.trainers import TRAINER_REGISTRY  # noqa: E402
from src.utils.data import build_task_bundles  # noqa: E402
from src.utils.runtime import move_batch_to_device, set_seed  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _indices_from_mask(mask: torch.Tensor) -> List[int]:
    return [int(i) for i, flag in enumerate(mask.tolist()) if bool(flag)]


def _collect_split_rows(
    *,
    model,
    dataloader,
    device,
    partition: NeuronPartition,
    decision_map: Sequence[int],
    true_task: int,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    stable_mask = partition.mask_for_role("stable")
    plastic_mask = partition.mask_for_role("shared") | partition.mask_for_role("reserve")
    stable_idx = _indices_from_mask(stable_mask)
    plastic_idx = _indices_from_mask(plastic_mask)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = move_batch_to_device(batch, device)
            for sample_i in range(int(targets.shape[0])):
                if max_samples is not None and len(rows) >= int(max_samples):
                    return rows
                pot, num_neurons = forward_potentials(model, inputs[sample_i])
                scores = aggregate_neuron_scores(pot, num_neurons)
                score_list = scores.detach().cpu().reshape(-1).tolist()
                natural = predict_wta_decision_map(scores, decision_map, allow_mask=None)
                stable_pred = predict_wta_decision_map(scores, decision_map, allow_mask=stable_mask)
                plastic_pred = predict_wta_decision_map(scores, decision_map, allow_mask=plastic_mask)
                true_y = int(targets[sample_i].item())
                oracle = stable_pred if int(true_task) == 0 else plastic_pred
                features = extract_role_path_features(
                    score_list, stable_indices=stable_idx, plastic_indices=plastic_idx
                )
                stable_correct = int(stable_pred) == true_y
                plastic_correct = int(plastic_pred) == true_y
                rows.append(
                    {
                        "true_task": int(true_task),
                        "true_class": true_y,
                        "natural_prediction": int(natural),
                        "stable_prediction": int(stable_pred),
                        "plastic_prediction": int(plastic_pred),
                        "oracle_prediction": int(oracle),
                        "stable_correct": bool(stable_correct),
                        "plastic_correct": bool(plastic_correct),
                        "features": features,
                    }
                )
    return rows


def collect_task_rows(
    *,
    model,
    trainer,
    task_bundles,
    config: Mapping[str, Any],
    device,
    partition: NeuronPartition,
    decision_map: Sequence[int],
    split: str,
    samples_per_task: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_id, bundle in enumerate(task_bundles[:2]):
        dataset = bundle.train_dataset if split == "validation" else bundle.test_dataset
        loader = trainer.build_eval_loader(dataset, config)
        rows.extend(
            _collect_split_rows(
                model=model,
                dataloader=loader,
                device=device,
                partition=partition,
                decision_map=decision_map,
                true_task=task_id,
                max_samples=samples_per_task,
            )
        )
    return rows


def run_eval(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    device: str,
    selector_seed: int,
    validation_samples_per_task: int,
    epochs: int,
    harmful_weight: float,
) -> Dict[str, Any]:
    set_seed(int(selector_seed))
    result = load_json(run_dir / "result.json")
    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()
    _, model, torch_device = load_checkpoint(run_dir, config, device, stage="task2")
    partition = NeuronPartition.from_role_payload(extract_partition_payload(result))
    if partition is None or not partition.enabled:
        raise ValueError("Neuron partition required.")
    decision_map = list(getattr(model, "decision_map"))
    task_bundles = build_task_bundles(config["data"], config["tasks"])
    if len(task_bundles) < 2:
        raise ValueError("Need two task bundles.")

    print("[collect] validation rows from train splits...", flush=True)
    validation_rows = collect_task_rows(
        model=model,
        trainer=trainer,
        task_bundles=task_bundles,
        config=config,
        device=torch_device,
        partition=partition,
        decision_map=decision_map,
        split="validation",
        samples_per_task=int(validation_samples_per_task),
    )
    print(f"[collect] validation n={len(validation_rows)}", flush=True)
    print("[collect] official test rows...", flush=True)
    test_rows = collect_task_rows(
        model=model,
        trainer=trainer,
        task_bundles=task_bundles,
        config=config,
        device=torch_device,
        partition=partition,
        decision_map=decision_map,
        split="test",
        samples_per_task=None,
    )
    print(f"[collect] test n={len(test_rows)}", flush=True)

    baselines = {
        "natural_wta": evaluate_fixed_path(test_rows, "natural"),
        "stable_only": evaluate_fixed_path(test_rows, "stable"),
        "plastic_only": evaluate_fixed_path(test_rows, "plastic"),
        "oracle_role_path": evaluate_fixed_path(test_rows, "oracle"),
        "margin_top1": evaluate_fixed_path(test_rows, "margin"),
    }

    fit_rows, calib_rows = stratified_disagree_split(validation_rows, seed=int(selector_seed), fit_fraction=0.7)
    fit_x = [row["features"] for row in fit_rows]
    fit_y = [int(role_path_label(bool(row["stable_correct"]), bool(row["plastic_correct"]))) for row in fit_rows]
    fit_w = [1.0] * len(fit_rows)
    selector = RolePathSelectorMLP(input_dim=len(ROLE_PATH_FEATURE_NAMES), seed=int(selector_seed))
    selector.fit(fit_x, fit_y, fit_w, epochs=int(epochs), harmful_weight=float(harmful_weight))
    calib_probs = selector.predict_plastic_probability([row["features"] for row in calib_rows])
    threshold_info = choose_probability_threshold(calib_rows, calib_probs)
    threshold = float(threshold_info["threshold"])

    test_probs = selector.predict_plastic_probability([row["features"] for row in test_rows])
    selector_metrics, _ = apply_role_path_policy(test_rows, probabilities=test_probs, threshold=threshold)
    val_probs = selector.predict_plastic_probability([row["features"] for row in validation_rows])
    val_metrics, _ = apply_role_path_policy(validation_rows, probabilities=val_probs, threshold=threshold)

    return {
        "system": "Winner-frequency roles + role-aware training + Single Role-path Selector",
        "run_dir": str(run_dir),
        "feature_names": list(ROLE_PATH_FEATURE_NAMES),
        "protocol": {
            "validation": f"first {validation_samples_per_task} eval-order train samples per task",
            "test": "official held-out test splits",
            "no_k8": True,
            "no_route_gate": True,
            "always_dual_subset_wta": True,
            "harmful_weight_on_stable_prefer_target": float(harmful_weight),
            "selector_seed": int(selector_seed),
            "epochs": int(epochs),
        },
        "selector": {
            "parameter_count": selector.parameter_count(),
            "threshold": threshold,
            "threshold_calibration": threshold_info,
            "fit_disagree_n": len(fit_rows),
            "calibration_disagree_n": len(calib_rows),
        },
        "validation": val_metrics,
        "test": {
            **baselines,
            "single_role_path_selector": selector_metrics,
        },
        "oracle_gap_pp": float(
            (baselines["oracle_role_path"]["macro_average"] - selector_metrics["macro_average"]) * 100.0
        ),
        "vs_natural_pp": float(
            (selector_metrics["macro_average"] - baselines["natural_wta"]["macro_average"]) * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selector-seed", type=int, default=0)
    parser.add_argument("--validation-samples-per-task", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--harmful-weight", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    config = load_config(run_dir, args.config)
    payload = run_eval(
        run_dir=run_dir,
        config=config,
        device=str(args.device),
        selector_seed=int(args.selector_seed),
        validation_samples_per_task=int(args.validation_samples_per_task),
        epochs=int(args.epochs),
        harmful_weight=float(args.harmful_weight),
    )
    out = resolve_path(args.output) if args.output else run_dir / "diagnostics" / "single_role_path_selector.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["test"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
