#!/usr/bin/env python3
"""
Unified entrypoint for continual-learning baseline reproduction runs.

This entrypoint focuses on orchestration while the trainer owns model details:

1. load a baseline config file,
2. validate the requested method,
3. prepare output folders,
4. dispatch to the matching baseline runner,
5. save a per-run JSON result,
6. append a compact CSV summary row.

The catastrophic-forgetting baseline is now dispatched to an official
SpykeTorch-backed trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers import TRAINER_REGISTRY


SUPPORTED_METHODS = {
    "catastrophic",
    "joint_training",
    "frozen_large_weights",
    "langevin",
}

# Summary CSV fields.
SUMMARY_FIELDS = [
    "timestamp",
    "run_name",
    "method",
    "seed",
    "task1_after_task1",
    "task1_after_task2",
    "task2_after_task2",
    "forgetting",
    "avg_acc",
    "notes",
    "result_json",
]


# CLI arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a continual-learning baseline from a config file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON or YAML config file under configs/baseline/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. Replaces config.seed if provided.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name override. Default is inferred from config + timestamp.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Optional device override. Replaces train.device if provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, print the execution plan, and stop before training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(raw_text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config support requires PyYAML. Install it or use JSON configs."
            ) from exc
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top of {path}")
        return data

    raise ValueError(
        f"Unsupported config format: {path.suffix}. Use .json, .yaml, or .yml."
    )


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_required_sections(config: Mapping[str, Any]) -> None:
    required_sections = ["method", "data", "tasks", "model", "train", "eval", "output"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")


def normalize_config(
    config: Dict[str, Any],
    config_path: Path,
    cli_seed: Optional[int],
    cli_run_name: Optional[str],
    cli_device: Optional[str],
) -> Dict[str, Any]:
    normalized = deepcopy(config)
    ensure_required_sections(normalized)

    method = str(normalized["method"]).strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: {sorted(SUPPORTED_METHODS)}"
        )

    if cli_seed is not None:
        normalized["seed"] = cli_seed
    normalized.setdefault("seed", 0)

    train = normalized.setdefault("train", {})
    if cli_device is not None:
        train["device"] = cli_device
    train.setdefault("device", "auto")

    output = normalized.setdefault("output", {})
    output.setdefault("root_dir", "experiments")
    output.setdefault("summary_csv", "results/baseline_summary.csv")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_run_name = f"{method}_seed{normalized['seed']}_{timestamp}"
    normalized["run_name"] = cli_run_name or normalized.get("run_name") or default_run_name
    normalized["config_path"] = str(config_path)
    return normalized


def resolve_output_paths(config: Mapping[str, Any], project_root: Path) -> Dict[str, Path]:
    output = config["output"]
    run_dir = project_root / output["root_dir"] / config["run_name"]
    results_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    summary_csv = project_root / output["summary_csv"]

    return {
        "run_dir": run_dir,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "summary_csv": summary_csv,
        "result_json": run_dir / "result.json",
        "resolved_config": run_dir / "resolved_config.json",
    }


def make_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class BaselineResult:
    method: str
    seed: int
    run_name: str
    metrics: Dict[str, Optional[float]]
    notes: str = ""
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "method": self.method,
            "seed": self.seed,
            "run_name": self.run_name,
            "metrics": self.metrics,
            "notes": self.notes,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


class BaseRunner:
    method_name = "base"

    def run(
        self,
        config: Mapping[str, Any],
        output_paths: Mapping[str, Path],
        dry_run: bool = False,
    ) -> BaselineResult:
        trainer_cls = TRAINER_REGISTRY.get(self.method_name)
        if trainer_cls is None:
            raise ValueError(f"No trainer registered for method '{self.method_name}'.")

        trainer = trainer_cls()
        trainer_result = trainer.run(config=config, dry_run=dry_run)
        return BaselineResult(
            method=self.method_name,
            seed=int(config["seed"]),
            run_name=str(config["run_name"]),
            metrics=trainer_result.metrics,
            notes=trainer_result.notes,
            extra={
                "validated_config": config,
                "output_paths": stringify_paths(output_paths),
                "trainer": trainer_result.extra,
            },
        )


class CatastrophicRunner(BaseRunner):
    method_name = "catastrophic"


class JointTrainingRunner(BaseRunner):
    method_name = "joint_training"


class FrozenLargeWeightsRunner(BaseRunner):
    method_name = "frozen_large_weights"


class LangevinRunner(BaseRunner):
    method_name = "langevin"


RUNNER_REGISTRY = {
    "catastrophic": CatastrophicRunner,
    "joint_training": JointTrainingRunner,
    "frozen_large_weights": FrozenLargeWeightsRunner,
    "langevin": LangevinRunner,
}


def stringify_paths(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    rendered: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Path):
            rendered[key] = str(value)
        else:
            rendered[key] = value
    return rendered


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_summary_row(
    summary_csv: Path,
    result: BaselineResult,
    result_json_path: Path,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_csv.exists()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": result.run_name,
        "method": result.method,
        "seed": result.seed,
        "task1_after_task1": result.metrics.get("task1_after_task1"),
        "task1_after_task2": result.metrics.get("task1_after_task2"),
        "task2_after_task2": result.metrics.get("task2_after_task2"),
        "forgetting": result.metrics.get("forgetting"),
        "avg_acc": result.metrics.get("avg_acc"),
        "notes": result.notes,
        "result_json": str(result_json_path),
    }

    with summary_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_execution_plan(config: Mapping[str, Any], output_paths: Mapping[str, Path]) -> None:
    preview = {
        "method": config["method"],
        "seed": config["seed"],
        "run_name": config["run_name"],
        "data": config["data"],
        "tasks": config["tasks"],
        "model": config["model"],
        "train": config["train"],
        "eval": config["eval"],
        "output_paths": stringify_paths(output_paths),
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    try:
        raw_config = load_config(config_path)
        config = normalize_config(
            raw_config,
            config_path,
            args.seed,
            args.run_name,
            args.device,
        )
        output_paths = resolve_output_paths(config, project_root)
        make_dirs(
            [
                output_paths["run_dir"],
                output_paths["results_dir"],
                output_paths["logs_dir"],
            ]
        )
        save_json(output_paths["resolved_config"], config)

        runner_cls = RUNNER_REGISTRY[config["method"]]
        runner = runner_cls()

        print_execution_plan(config, output_paths)
        result = runner.run(config, output_paths, dry_run=args.dry_run)

        result_payload = result.to_dict()
        result_payload["config_path"] = config["config_path"]
        result_payload["resolved_config_path"] = str(output_paths["resolved_config"])
        save_json(output_paths["result_json"], result_payload)
        append_summary_row(output_paths["summary_csv"], result, output_paths["result_json"])

        print(f"Saved result to {output_paths['result_json']}")
        print(f"Updated summary at {output_paths['summary_csv']}")
        return 0
    except Exception as exc:
        print(f"[run_baseline] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




