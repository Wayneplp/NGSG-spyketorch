from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from SpykeTorch import snn

from src.analysis.metrics import summarize_continual_metrics
from src.plasticity import SpykeTorchRSTDPConfig, SpykeTorchRewardSTDP
from src.utils.data import (
    ConcatenatedSubset,
    TaskBundle,
    build_dataloader,
    build_task_bundles,
    bundle_summary,
)
from src.utils.runtime import move_batch_to_device, set_seed


@dataclass
class TrainerResult:
    metrics: Dict[str, Optional[float]]
    notes: str = ""
    extra: Optional[Dict[str, Any]] = None


class BaselineTrainer:
    """Catastrophic-forgetting trainer backed by the official SpykeTorch package."""

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name

    def run(self, config: Mapping[str, Any], dry_run: bool = False) -> TrainerResult:
        if dry_run:
            bundles = self._try_describe_tasks(config)
            return TrainerResult(
                metrics=self.empty_metrics(),
                notes=f"Dry run for baseline '{self.method_name}'.",
                extra={
                    "trainer_plan": self.describe_plan(config),
                    "task_summary": bundles,
                },
            )

        self._ensure_runtime_dependencies()
        set_seed(int(config.get("seed", 0)))

        device = self.resolve_device(config)
        task_bundles = build_task_bundles(config["data"], config["tasks"])
        if len(task_bundles) < 2:
            raise ValueError("Expected at least two task bundles for continual-learning baselines.")

        task1, task2 = task_bundles[0], task_bundles[1]
        model = self.build_model(config).to(device)
        rstdp = self.build_output_rstdp(model, config).to(device)

        train_task1_loader = self.build_train_loader(task1.train_dataset, config)
        test_task1_loader = self.build_eval_loader(task1.test_dataset, config)
        test_task2_loader = self.build_eval_loader(task2.test_dataset, config)

        task1_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task1_loader,
            config=config,
            rstdp=rstdp,
            device=device,
            stage_name="task1",
        )
        task1_after_task1 = self.evaluate(model, test_task1_loader, device)

        train_task2_loader = self.build_task2_train_loader(task1, task2, config)
        task2_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task2_loader,
            config=config,
            rstdp=rstdp,
            device=device,
            stage_name="task2",
        )

        task1_after_task2 = self.evaluate(model, test_task1_loader, device)
        task2_after_task2 = self.evaluate(model, test_task2_loader, device)
        metrics = summarize_continual_metrics(
            task1_after_task1=task1_after_task1,
            task1_after_task2=task1_after_task2,
            task2_after_task2=task2_after_task2,
        )

        return TrainerResult(
            metrics=metrics,
            notes=self.implementation_note(config),
            extra={
                "device": str(device),
                "task_summary": bundle_summary(task_bundles),
                "trainer_plan": self.describe_plan(config),
                "task1_training": task1_training_stats,
                "task2_training": task2_training_stats,
                "model_summary": self.summarize_model(model),
            },
        )

    def resolve_device(self, config: Mapping[str, Any]) -> torch.device:
        requested = str(config.get("train", {}).get("device", "auto")).lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Config requested CUDA but no CUDA device is available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, config: Mapping[str, Any]) -> nn.Module:
        model_module = import_module("src.models")
        model_overrides = dict(config.get("model", {}))
        architecture = str(model_overrides.pop("architecture", "spyketorch")).lower()
        if architecture not in {"spyketorch", "official_spyketorch"}:
            raise ValueError(
                "The main src/ implementation now only supports architecture='spyketorch'. "
                "Legacy approximations live under approx/legacy_approx/."
            )
        build_baseline_network = getattr(model_module, "build_baseline_network")
        return build_baseline_network(model_overrides)

    def build_output_rstdp(self, model: nn.Module, config: Mapping[str, Any]) -> SpykeTorchRewardSTDP:
        train_cfg = config.get("train", {})
        return SpykeTorchRewardSTDP(
            model.s3,
            SpykeTorchRSTDPConfig(
                reward_active=float(train_cfg.get("reward_active", 0.004)),
                reward_inactive=float(train_cfg.get("reward_inactive", -0.003)),
                punish_active=float(train_cfg.get("punish_active", -0.003)),
                punish_inactive=float(train_cfg.get("punish_inactive", 0.0005)),
                reward_scale=float(train_cfg.get("reward_scale", 1.0)),
                punish_scale=float(train_cfg.get("punish_scale", 1.0)),
                lower_bound=float(train_cfg.get("weight_clip_min", 0.0)),
                upper_bound=float(train_cfg.get("weight_clip_max", 1.0)),
                use_stabilizer=bool(train_cfg.get("use_weight_stabilizer", True)),
            ),
        )

    def build_train_loader(self, dataset: Any, config: Mapping[str, Any]) -> Any:
        train_cfg = config.get("train", {})
        return build_dataloader(
            dataset=dataset,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=bool(train_cfg.get("shuffle", True)),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )

    def build_eval_loader(self, dataset: Any, config: Mapping[str, Any]) -> Any:
        eval_cfg = config.get("eval", {})
        return build_dataloader(
            dataset=dataset,
            batch_size=int(eval_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", 0)),
        )

    def build_task2_train_loader(
        self,
        task1: TaskBundle,
        task2: TaskBundle,
        config: Mapping[str, Any],
    ) -> Any:
        return self.build_train_loader(task2.train_dataset, config)

    def train_single_task(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        rstdp: SpykeTorchRewardSTDP,
        device: torch.device,
        stage_name: str,
    ) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        learning_rule = str(train_cfg.get("learning_rule", "spyketorch_stdp_rstdp")).lower()
        if learning_rule not in {"spyketorch_stdp_rstdp", "spyketorch", "paper_stdp_rstdp"}:
            raise ValueError(
                "Use learning_rule='spyketorch_stdp_rstdp' for the main implementation. "
                "Legacy approximation configs are stored under approx/legacy_approx/."
            )

        s1_epochs = self._stage_epochs(config, stage_name, "s1_stdp_epochs", 0)
        s2_epochs = self._stage_epochs(config, stage_name, "s2_stdp_epochs", 0)
        s3_epochs = int(train_cfg.get(f"num_epochs_{stage_name}", train_cfg.get("num_epochs", 1)))

        stats: Dict[str, Any] = {
            "stage": stage_name,
            "learning_rule": "official_spyketorch_stdp_anti_stdp",
            "feature_training": {},
        }

        if s1_epochs > 0:
            stats["feature_training"]["s1"] = self.train_s1_stdp(model, dataloader, config, device, s1_epochs)
        if s2_epochs > 0:
            stats["feature_training"]["s2"] = self.train_s2_stdp(model, dataloader, config, device, s2_epochs)
        stats["output_training"] = self.train_s3_rstdp(model, dataloader, config, rstdp, device, s3_epochs)
        return stats

    def train_s1_stdp(self, model: nn.Module, dataloader: Any, config: Mapping[str, Any], device: torch.device, epochs: int) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        stdp = snn.STDP(
            model.s1,
            (float(train_cfg.get("stdp_a_plus", 0.004)), float(train_cfg.get("stdp_a_minus", -0.003))),
            use_stabilizer=bool(train_cfg.get("use_weight_stabilizer", True)),
            lower_bound=float(train_cfg.get("weight_clip_min", 0.0)),
            upper_bound=float(train_cfg.get("weight_clip_max", 1.0)),
        )
        stdp.to(device)
        kwta = int(train_cfg.get("stdp_kwta", 1))
        radius = int(train_cfg.get("s1_inhibition_radius", 0))
        history = []
        for epoch_idx in range(epochs):
            samples = 0
            for image, _ in self.iter_samples(dataloader, device):
                encoded = model.encode(image)
                s1 = model.s1_step(encoded)
                stdp(encoded, s1["potentials"], s1["spikes"], kwta=kwta, inhibition_radius=radius)
                samples += 1
            history.append({"epoch": epoch_idx + 1, "samples": samples, "stdp_updates": samples})
        return {"layer": "s1", "epochs": epochs, "history": history}

    def train_s2_stdp(self, model: nn.Module, dataloader: Any, config: Mapping[str, Any], device: torch.device, epochs: int) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        stdp = snn.STDP(
            model.s2,
            (float(train_cfg.get("stdp_a_plus", 0.004)), float(train_cfg.get("stdp_a_minus", -0.003))),
            use_stabilizer=bool(train_cfg.get("use_weight_stabilizer", True)),
            lower_bound=float(train_cfg.get("weight_clip_min", 0.0)),
            upper_bound=float(train_cfg.get("weight_clip_max", 1.0)),
        )
        stdp.to(device)
        kwta = int(train_cfg.get("stdp_kwta", 1))
        radius = int(train_cfg.get("s2_inhibition_radius", 0))
        history = []
        for epoch_idx in range(epochs):
            samples = 0
            for image, _ in self.iter_samples(dataloader, device):
                encoded = model.encode(image)
                s1 = model.s1_step(encoded)
                s2 = model.s2_step(s1["pooled"])
                stdp(s1["pooled"], s2["potentials"], s2["spikes"], kwta=kwta, inhibition_radius=radius)
                samples += 1
            history.append({"epoch": epoch_idx + 1, "samples": samples, "stdp_updates": samples})
        return {"layer": "s2", "epochs": epochs, "history": history}

    def train_s3_rstdp(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        rstdp: SpykeTorchRewardSTDP,
        device: torch.device,
        epochs: int,
    ) -> Dict[str, Any]:
        num_classes = int(config["model"]["num_classes"])
        neurons_per_class = int(config["model"]["neurons_per_class"])
        train_cfg = config.get("train", {})
        kwta = int(train_cfg.get("rstdp_kwta", train_cfg.get("s3_kwta", 1)))
        radius = int(train_cfg.get("s3_inhibition_radius", 0))
        history = []
        for epoch_idx in range(epochs):
            samples = 0
            correct = 0
            reward_updates = 0
            punish_updates = 0
            for image, target in self.iter_samples(dataloader, device):
                features = model.forward_spikes(image)
                update_stats = rstdp.update(
                    input_spikes=features["s3"]["input"],
                    potentials=features["s3"]["potentials"],
                    output_spikes=features["s3"]["spikes"],
                    target=int(target.item()),
                    num_classes=num_classes,
                    neurons_per_class=neurons_per_class,
                    kwta=kwta,
                    inhibition_radius=radius,
                )
                samples += 1
                correct += int(update_stats["prediction"] == int(target.item()))
                reward_updates += int(update_stats["reward_updates"])
                punish_updates += int(update_stats["punish_updates"])
            history.append(
                {
                    "epoch": epoch_idx + 1,
                    "samples": samples,
                    "train_acc_proxy": float(correct / max(samples, 1)),
                    "reward_updates": reward_updates,
                    "punish_updates": punish_updates,
                }
            )
        return {"stage": "s3", "epochs": epochs, "learning_rule": "spyketorch_stdp_anti_stdp", "history": history}

    def iter_samples(self, dataloader: Any, device: torch.device):
        for batch in dataloader:
            inputs, targets = move_batch_to_device(batch, device)
            for sample_idx in range(int(targets.shape[0])):
                yield inputs[sample_idx], targets[sample_idx]

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: Any, device: torch.device) -> float:
        correct = 0
        total = 0
        for image, target in self.iter_samples(dataloader, device):
            prediction = model.predict_single(image)
            correct += int(prediction == int(target.item()))
            total += 1
        return float(correct / max(total, 1))

    def _stage_epochs(self, config: Mapping[str, Any], stage_name: str, key: str, default: int) -> int:
        train_cfg = config.get("train", {})
        stage_key = f"{key}_{stage_name}"
        return int(train_cfg.get(stage_key, train_cfg.get(key, default)))

    def describe_plan(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "dataset": config.get("data", {}),
            "tasks": config.get("tasks", {}),
            "train": config.get("train", {}),
            "eval": config.get("eval", {}),
            "implementation": "official SpykeTorch package",
        }

    def summarize_model(self, model: nn.Module) -> Dict[str, Any]:
        if hasattr(model, "describe"):
            return model.describe()
        return {"repr": repr(model)}

    def empty_metrics(self) -> Dict[str, Optional[float]]:
        return {
            "task1_after_task1": None,
            "task1_after_task2": None,
            "task2_after_task2": None,
            "forgetting": None,
            "avg_acc": None,
        }

    def implementation_note(self, config: Optional[Mapping[str, Any]] = None) -> str:
        return (
            "Official SpykeTorch-based baseline: S1/S2 use SpykeTorch snn.STDP, "
            "S1/S2/S3 layers are SpykeTorch snn.Convolution/Pooling modules, and S3 "
            "uses the SpykeTorch tutorial-style rule: official snn.STDP for correct "
            "winners and official snn.STDP anti-STDP for wrong winners."
        )

    def _try_describe_tasks(self, config: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        try:
            return bundle_summary(build_task_bundles(config["data"], config["tasks"]))
        except Exception:
            task_names = list(config.get("tasks", {}).get("task_names", []))
            task_splits = list(config.get("tasks", {}).get("task_splits", []))
            return [{"name": str(name), "labels": labels} for name, labels in zip(task_names, task_splits)]

    def _ensure_runtime_dependencies(self) -> None:
        try:
            import_module("SpykeTorch")
            import_module("torchvision")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The main implementation requires the official SpykeTorch package. "
                "Install it with: pip install git+https://github.com/miladmozafari/SpykeTorch.git"
            ) from exc


class CatastrophicForgettingTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="catastrophic")


class JointTrainingTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="joint_training")

    def build_task2_train_loader(self, task1: TaskBundle, task2: TaskBundle, config: Mapping[str, Any]) -> Any:
        joint_dataset = ConcatenatedSubset([task1.train_dataset, task2.train_dataset])
        return self.build_train_loader(joint_dataset, config)


class FrozenLargeWeightsTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="frozen_large_weights")


class LangevinTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="langevin")


TRAINER_REGISTRY = {
    "catastrophic": CatastrophicForgettingTrainer,
    "joint_training": JointTrainingTrainer,
    "frozen_large_weights": FrozenLargeWeightsTrainer,
    "langevin": LangevinTrainer,
}





