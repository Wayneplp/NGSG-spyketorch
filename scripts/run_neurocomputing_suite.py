#!/usr/bin/env python3
"""Run one Neurocomputing matrix cell: train (+ optional K8/Gain/Oracle post-hoc)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_neurocomputing_configs import METHODS, ORDERS, generate  # noqa: E402


def run_cmd(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=True)


def config_path(order: str, method: str, seed: int) -> Path:
    return PROJECT_ROOT / "configs" / "neurocomputing" / f"neurocomputing_{order}_{method}_seed{seed}.yaml"


def run_dir_for(order: str, method: str, seed: int) -> Path:
    return PROJECT_ROOT / "experiments" / f"neurocomputing_{order}_{method}_seed{seed}"


def ensure_configs(orders: Sequence[str], methods: Sequence[str], seeds: Sequence[int]) -> None:
    missing = [
        config_path(order, method, seed)
        for order in orders
        for method in methods
        for seed in seeds
        if not config_path(order, method, seed).exists()
    ]
    if missing:
        generate(seeds, orders, methods)


def train_one(order: str, method: str, seed: int, device: str, dry_run: bool) -> Path:
    cfg = config_path(order, method, seed)
    if not cfg.exists():
        generate([seed], [order], [method])
    cmd = [
        sys.executable,
        "scripts/run_baseline.py",
        "--config",
        str(cfg.relative_to(PROJECT_ROOT)),
        "--device",
        device,
    ]
    if dry_run:
        cmd.append("--dry-run")
        run_cmd(cmd, dry_run=False)
    else:
        run_cmd(cmd, dry_run=False)
    return run_dir_for(order, method, seed)


def posthoc_role_pipeline(run_dir: Path, seed: int, device: str, dry_run: bool) -> Dict[str, Path]:
    diag = run_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    k8_json = diag / "k8_switch_selector_final.json"
    gain_json = diag / "k8_bidirectional_gain_selector.json"

    run_cmd(
        [
            sys.executable,
            "scripts/eval_k8_switch_selector.py",
            "--run-dir",
            str(run_dir.relative_to(PROJECT_ROOT)),
            "--checkpoint-stage",
            "task2",
            "--device",
            device,
            "--k",
            "8",
            "--prototype-samples-per-task",
            "2400",
            "--validation-samples-per-task",
            "2000",
            "--selector-seed",
            str(seed),
            "--output-json",
            str(k8_json.relative_to(PROJECT_ROOT)),
            "--write-markdown",
        ],
        dry_run=dry_run,
    )

    val_jsonl = k8_json.with_name(f"{k8_json.stem}_validation_records.jsonl")
    test_jsonl = k8_json.with_name(f"{k8_json.stem}_test_records.jsonl")
    run_cmd(
        [
            sys.executable,
            "scripts/train_k8_bidirectional_selectors.py",
            "--validation-jsonl",
            str(val_jsonl.relative_to(PROJECT_ROOT)),
            "--test-jsonl",
            str(test_jsonl.relative_to(PROJECT_ROOT)),
            "--output-json",
            str(gain_json.relative_to(PROJECT_ROOT)),
            "--select-route-threshold-from-validation",
            "--gain-threshold",
            "0.0",
            "--selector-seed",
            str(seed),
            "--beneficial-weight",
            "1.0",
            "--harmful-weight",
            "3.0",
            "--neutral-weight",
            "0.25",
        ],
        dry_run=dry_run,
    )
    return {"k8": k8_json, "gain": gain_json}


def summarize_cell(order: str, method: str, seed: int) -> Dict[str, Any]:
    run_dir = run_dir_for(order, method, seed)
    result_path = run_dir / "result.json"
    payload: Dict[str, Any] = {
        "order": order,
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir),
        "result_present": result_path.exists(),
    }
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        payload["metrics"] = result.get("metrics")
    gain_path = run_dir / "diagnostics" / "k8_bidirectional_gain_selector.json"
    k8_path = run_dir / "diagnostics" / "k8_switch_selector_final.json"
    if k8_path.exists():
        k8 = json.loads(k8_path.read_text(encoding="utf-8"))
        payload["k8"] = k8.get("comparison", {})
    if gain_path.exists():
        gain = json.loads(gain_path.read_text(encoding="utf-8"))
        payload["gain"] = {
            "route_threshold": gain.get("protocol", {}).get("route_threshold"),
            "test": gain.get("test"),
            "validation": gain.get("validation"),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", default=["mnist_to_emnist"], choices=list(ORDERS))
    parser.add_argument("--methods", nargs="+", default=["role_train"], choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true")
    parser.add_argument("--posthoc-only-role-train", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default="results/neurocomputing_matrix_status.json")
    args = parser.parse_args()

    ensure_configs(args.orders, args.methods, args.seeds)
    summaries: List[Dict[str, Any]] = []
    for order in args.orders:
        for method in args.methods:
            for seed in args.seeds:
                print(f"\n=== {order} | {method} | seed={seed} ===", flush=True)
                if not args.skip_train:
                    train_one(order, method, seed, args.device, args.dry_run)
                if not args.skip_posthoc and method == "role_train":
                    posthoc_role_pipeline(
                        run_dir_for(order, method, seed),
                        int(seed),
                        args.device,
                        args.dry_run,
                    )
                if not args.dry_run:
                    summaries.append(summarize_cell(order, method, seed))

    out = PROJECT_ROOT / args.summary_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote status summary to {out}")


if __name__ == "__main__":
    main()
