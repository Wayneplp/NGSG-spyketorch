#!/usr/bin/env python3
"""Compare inference readout strategies on a saved Task2 model checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.readout import (  # noqa: E402
    ALL_STRATEGIES,
    NeuronLabelMaps,
    ReadoutStrategy,
    STRATEGY_LABELS,
    aggregate_neuron_scores,
    extract_winner_label_counts,
    predict_from_scores,
)
from src.continual.reserve_activation import _resolve_num_s3_neurons, potential_planes  # noqa: E402
from src.trainers import TRAINER_REGISTRY  # noqa: E402
from src.utils.data import build_task_bundles  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_checkpoint(run_dir: Path, config: Mapping[str, Any], device):
    import torch

    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()
    torch_device = trainer.resolve_device({"train": {"device": device}})
    model = trainer.build_model(config).to(torch_device)

    model_path = run_dir / "artifacts" / "model_after_task2.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    payload = torch.load(model_path, map_location=torch_device)
    model.load_state_dict(payload["model_state_dict"])
    if payload.get("decision_map") is not None:
        model.decision_map = payload["decision_map"]
    model.eval()
    return trainer, model, torch_device


def _sample_uses_s3_cache(sample) -> bool:
    import torch

    if not isinstance(sample, torch.Tensor):
        return False
    # Cached C2/S3-input tensors are 3D/4D/5D feature maps, not 28x28 images.
    if sample.ndim == 3:
        return int(sample.shape[0]) > 16
    if sample.ndim == 4:
        return int(sample.shape[1]) > 16 or int(sample.shape[0]) > 16
    if sample.ndim == 5:
        return True
    return False


def forward_potentials(model, sample):
    import torch

    with torch.no_grad():
        if _sample_uses_s3_cache(sample):
            _ = model.forward_from_s3_input(sample)
            pot = getattr(model, "ctx", {}).get("potentials")
        elif hasattr(model, "forward_s3_potentials"):
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
        raise RuntimeError("Model context missing potentials after forward.")
    num_neurons = _resolve_num_s3_neurons(model, pot)
    return pot, num_neurons


def evaluate_readouts(
    model,
    dataloader,
    device,
    *,
    strategies: Iterable[ReadoutStrategy],
    label_maps: NeuronLabelMaps,
    neurons_per_class: int,
    eval_task: str,
) -> Dict[str, Dict[str, float]]:
    from src.trainers.baseline_trainer import move_batch_to_device

    totals: Dict[str, int] = {strategy.value: 0 for strategy in strategies}
    correct: Dict[str, int] = {strategy.value: 0 for strategy in strategies}
    silent: Dict[str, int] = {strategy.value: 0 for strategy in strategies}

    for batch in dataloader:
        inputs, targets = move_batch_to_device(batch, device)
        for sample_idx in range(int(targets.shape[0])):
            target = int(targets[sample_idx].item())
            pot, num_neurons = forward_potentials(model, inputs[sample_idx])
            scores = aggregate_neuron_scores(pot, num_neurons)
            for strategy in strategies:
                prediction = predict_from_scores(
                    scores,
                    strategy=strategy,
                    label_maps=label_maps,
                    neurons_per_class=neurons_per_class,
                    eval_task=eval_task,
                )
                key = strategy.value
                totals[key] += 1
                if prediction < 0:
                    silent[key] += 1
                elif prediction == target:
                    correct[key] += 1
    out: Dict[str, Dict[str, float]] = {}
    for strategy in strategies:
        key = strategy.value
        denom = max(totals[key], 1)
        out[key] = {
            "accuracy": float(correct[key] / denom),
            "correct": float(correct[key]),
            "total": float(totals[key]),
            "silent": float(silent[key]),
            "label": STRATEGY_LABELS[strategy],
        }
    return out


def build_label_maps(result: Mapping[str, Any], model) -> NeuronLabelMaps:
    decision_map = getattr(model, "decision_map", None)
    if decision_map is None:
        raise ValueError("Model is missing decision_map.")
    num_classes = int((result.get("extra") or {}).get("trainer", {}).get("num_classes") or 10)
    if num_classes <= 0:
        num_classes = 10
    return NeuronLabelMaps.from_training_stats(
        decision_map=decision_map,
        winner_label_counts_task1=extract_winner_label_counts(result, "task1"),
        winner_label_counts_task2=extract_winner_label_counts(result, "task2"),
        num_classes=num_classes,
    )


def format_table(rows: Mapping[str, Mapping[str, Any]], *, baseline_key: str) -> str:
    baseline_acc = float(rows[baseline_key]["accuracy"])
    lines = [
        "| strategy | accuracy | delta vs baseline | silent |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, payload in rows.items():
        acc = float(payload["accuracy"])
        delta_pp = (acc - baseline_acc) * 100.0
        lines.append(
            f"| {payload['label']} | {acc * 100:.2f}% | {delta_pp:+.2f} pp | {int(payload['silent'])} |"
        )
    return "\n".join(lines)


def run_ablation(run_dir: Path, device: str) -> Dict[str, Any]:
    result_path = run_dir / "result.json"
    config_path = run_dir / "resolved_config.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.json: {result_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved_config.json: {config_path}")

    result = load_json(result_path)
    config = load_json(config_path)
    trainer, model, torch_device = load_model_checkpoint(run_dir, config, device)
    label_maps = build_label_maps(result, model)
    neurons_per_class = int(config.get("model", {}).get("neurons_per_class", 20))

    bundles = build_task_bundles(config["data"], config["tasks"])
    task1, task2 = bundles[0], bundles[1]
    test_task1_loader = trainer.build_eval_loader(task1.test_dataset, config)
    test_task2_loader = trainer.build_eval_loader(task2.test_dataset, config)

    metrics = result.get("metrics") or {}
    embedded = {
        "task1_test_after_task2": float(metrics.get("task1_after_task2", 0.0)),
        "task2_test_after_task2": float(metrics.get("task2_after_task2", 0.0)),
    }

    task1_rows = evaluate_readouts(
        model,
        test_task1_loader,
        torch_device,
        strategies=ALL_STRATEGIES,
        label_maps=label_maps,
        neurons_per_class=neurons_per_class,
        eval_task="task1",
    )
    task2_rows = evaluate_readouts(
        model,
        test_task2_loader,
        torch_device,
        strategies=ALL_STRATEGIES,
        label_maps=label_maps,
        neurons_per_class=neurons_per_class,
        eval_task="task2",
    )

    summary = {
        "run_name": result.get("run_name"),
        "run_dir": str(run_dir),
        "label_mismatch": label_maps.mismatch_counts(),
        "embedded_metrics": embedded,
        "task1_test_after_task2": task1_rows,
        "task2_test_after_task2": task2_rows,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment directory with artifacts/.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--write-json", type=Path, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    summary = run_ablation(run_dir, args.device)

    print(f"# Readout ablation: {summary['run_name']}")
    print(f"- run_dir: {summary['run_dir']}")
    print(f"- label mismatch vs static map: {summary['label_mismatch']}")
    print(
        "- embedded result.json metrics: "
        f"task1_after_task2={summary['embedded_metrics']['task1_test_after_task2']:.4f} "
        f"task2_after_task2={summary['embedded_metrics']['task2_test_after_task2']:.4f}"
    )
    print("\n## MNIST test after Task2")
    print(format_table(summary["task1_test_after_task2"], baseline_key=ReadoutStrategy.WTA_STATIC.value))
    print("\n## EMNIST test after Task2")
    print(format_table(summary["task2_test_after_task2"], baseline_key=ReadoutStrategy.WTA_STATIC.value))

    best_t1 = max(summary["task1_test_after_task2"].items(), key=lambda item: item[1]["accuracy"])
    best_t2 = max(summary["task2_test_after_task2"].items(), key=lambda item: item[1]["accuracy"])
    base_t1 = summary["task1_test_after_task2"][ReadoutStrategy.WTA_STATIC.value]["accuracy"]
    base_t2 = summary["task2_test_after_task2"][ReadoutStrategy.WTA_STATIC.value]["accuracy"]
    print("\n## Best vs baseline")
    print(
        f"- MNIST: {best_t1[1]['label']} -> {best_t1[1]['accuracy'] * 100:.2f}% "
        f"({(best_t1[1]['accuracy'] - base_t1) * 100:+.2f} pp vs current readout)"
    )
    print(
        f"- EMNIST: {best_t2[1]['label']} -> {best_t2[1]['accuracy'] * 100:.2f}% "
        f"({(best_t2[1]['accuracy'] - base_t2) * 100:+.2f} pp vs current readout)"
    )

    if args.write_json:
        out_path = args.write_json if args.write_json.is_absolute() else PROJECT_ROOT / args.write_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
