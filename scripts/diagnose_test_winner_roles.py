#!/usr/bin/env python3
"""Print or recompute Task2 test-time natural WTA winner role distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.continual.neuron_partition import NeuronPartition
from src.trainers import TRAINER_REGISTRY


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_partition_payload(result: Mapping[str, Any]) -> Dict[str, Any]:
    trainer = (result.get("extra") or {}).get("trainer") or {}
    task1 = trainer.get("task1_training") or {}
    partition = task1.get("neuron_partition")
    if isinstance(partition, dict) and partition.get("roles"):
        return partition
    summary = trainer.get("neuron_partition")
    if isinstance(summary, dict) and summary.get("role_counts"):
        raise ValueError(
            "result.json only has neuron_partition summary without per-neuron roles. "
            "Re-run with updated trainer or pass --run-dir containing task1 partition arrays."
        )
    raise ValueError("Could not find neuron_partition.roles in result.json.")


def extract_test_winner_roles(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    extra = result.get("extra") or {}
    roles = extra.get("test_winner_roles")
    if isinstance(roles, dict):
        return roles
    trainer = extra.get("trainer") or {}
    roles = trainer.get("test_winner_roles")
    return roles if isinstance(roles, dict) else None


def format_role_block(label: str, payload: Mapping[str, Any]) -> str:
    lines = [
        f"## {label}",
        f"- accuracy: {payload.get('accuracy', 0.0):.4f}",
        f"- reserve test win rate: {payload.get('reserve_test_win_rate', 0.0):.4f}",
        f"- stable winner wrong (decision!=target): {payload.get('stable_winner_wrong_fraction', 0.0):.4f}",
        "",
        "| role | wins | win fraction | role accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    role_wins = payload.get("role_wins") or {}
    role_fracs = payload.get("role_win_fractions") or {}
    role_acc = payload.get("role_accuracy") or {}
    for role in ("stable", "shared", "reserve", "dead"):
        lines.append(
            f"| {role} | {role_wins.get(role, 0)} | "
            f"{role_fracs.get(role, 0.0):.4f} | {role_acc.get(role, 0.0):.4f} |"
        )
    return "\n".join(lines)


def print_report(result: Mapping[str, Any], *, source: str) -> None:
    metrics = result.get("metrics") or {}
    reserve = ((result.get("extra") or {}).get("reserve_activation") or {})
    test_roles = extract_test_winner_roles(result)

    print(f"# Test-time winner role report ({source})")
    print(f"- run_name: {result.get('run_name')}")
    print(
        "- metrics: "
        f"task1_after_task2={metrics.get('task1_after_task2')} "
        f"task2_after_task2={metrics.get('task2_after_task2')} "
        f"avg_acc={metrics.get('avg_acc')}"
    )
    if reserve.get("enabled"):
        print(
            "- reserve train stats: "
            f"recruitment_rate={reserve.get('recruitment_rate')} "
            f"recruit_condition={reserve.get('recruit_condition', reserve.get('config', {}).get('recruit_condition'))}"
        )
        skip_stats = reserve.get("skip_stats") or {}
        if skip_stats:
            print(f"- reserve skip_stats: {skip_stats}")

    if test_roles is None:
        print(
            "\nNo test_winner_roles in result.json. "
            "Re-run the experiment with the updated trainer, or use --re-evaluate."
        )
        return

    for key in ("task1_test_after_task2", "task2_test_after_task2"):
        block = test_roles.get(key)
        if isinstance(block, dict):
            print()
            print(format_role_block(key, block))

    task2 = test_roles.get("task2_test_after_task2") or {}
    reserve_rate = float(task2.get("reserve_test_win_rate", 0.0))
    train_rate = float(reserve.get("recruitment_rate", 0.0) or 0.0)
    print("\n## Diagnosis")
    if train_rate > 0.2 and reserve_rate < 0.1:
        print(
            f"- Train recruitment ({train_rate:.1%}) >> test reserve win rate ({reserve_rate:.1%}). "
            "Supports STDP reroute vs natural-WTA inference mismatch."
        )
    elif reserve_rate >= 0.1:
        print(
            f"- Reserve neurons win {reserve_rate:.1%} of test samples; "
            "reroute may partially reach inference, but accuracy may still be low."
        )
    else:
        print("- Reserve test win rate is very low; check partition or model checkpoint.")


def re_evaluate(run_dir: Path, device: str) -> Dict[str, Any]:
    result_path = run_dir / "result.json"
    config_path = run_dir / "resolved_config.json"
    model_path = run_dir / "artifacts" / "model_after_task2.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}. "
            "Re-run with output.save_task2_model: true in config."
        )

    config = load_json(config_path)
    result = load_json(result_path) if result_path.exists() else {}
    partition = NeuronPartition.from_role_payload(extract_partition_payload(result))

    trainer_cls = TRAINER_REGISTRY.get(str(config.get("method", "catastrophic")))
    if trainer_cls is None:
        raise ValueError(f"No trainer for method={config.get('method')}")
    trainer = trainer_cls()

    import torch

    torch_device = trainer.resolve_device({"train": {"device": device}})
    task_bundles = trainer.build_task_bundles_from_config(config) if hasattr(trainer, "build_task_bundles_from_config") else None
    from src.utils.data import build_task_bundles

    bundles = build_task_bundles(config["data"], config["tasks"])
    task1, task2 = bundles[0], bundles[1]
    model = trainer.build_model(config).to(torch_device)
    payload = torch.load(model_path, map_location=torch_device)
    model.load_state_dict(payload["model_state_dict"])
    if payload.get("decision_map") is not None:
        model.decision_map = payload["decision_map"]
    model.eval()

    test_task1_loader = trainer.build_eval_loader(task1.test_dataset, config)
    test_task2_loader = trainer.build_eval_loader(task2.test_dataset, config)
    test_roles = {
        "task1_test_after_task2": trainer.evaluate_winner_role_distribution(
            model,
            test_task1_loader,
            torch_device,
            partition,
            stage_label="task1_test_after_task2",
        ),
        "task2_test_after_task2": trainer.evaluate_winner_role_distribution(
            model,
            test_task2_loader,
            torch_device,
            partition,
            stage_label="task2_test_after_task2",
        ),
    }

    out_path = run_dir / "diagnostics" / "test_winner_roles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(test_roles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result_path.exists():
        merged = dict(result)
        extra = dict(merged.get("extra") or {})
        extra["test_winner_roles"] = test_roles
        merged["extra"] = extra
        result_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return test_roles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Path to experiments/<run>/result.json (prints embedded test_winner_roles).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Experiment run directory; use with --re-evaluate to recompute from saved model.",
    )
    parser.add_argument(
        "--re-evaluate",
        action="store_true",
        help="Recompute test_winner_roles from artifacts/model_after_task2.pt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for --re-evaluate.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional path to write a compact JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.re_evaluate:
        if args.run_dir is None:
            print("error: --re-evaluate requires --run-dir", file=sys.stderr)
            return 2
        run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
        test_roles = re_evaluate(run_dir, args.device)
        result = load_json(run_dir / "result.json")
        result.setdefault("extra", {})["test_winner_roles"] = test_roles
        print_report(result, source=f"re-evaluated from {run_dir}")
        if args.write_json:
            args.write_json.write_text(json.dumps(test_roles, indent=2) + "\n", encoding="utf-8")
        return 0

    result_path = args.result_json
    if result_path is None and args.run_dir is not None:
        run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
        result_path = run_dir / "result.json"
    if result_path is None:
        print("error: provide --result-json or --run-dir", file=sys.stderr)
        return 2
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    if not result_path.exists():
        print(f"error: file not found: {result_path}", file=sys.stderr)
        return 1

    result = load_json(result_path)
    print_report(result, source=str(result_path))
    if args.write_json:
        roles = extract_test_winner_roles(result)
        if roles is not None:
            args.write_json.write_text(json.dumps(roles, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
