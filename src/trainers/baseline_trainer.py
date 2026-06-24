from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from src.analysis.metrics import summarize_continual_metrics
from src.plasticity.rstdp import RSTDPConfig, RewardModulatedSTDP
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
    """
    First runnable continual-learning trainer.

    This is intentionally a minimal approximation:
    - shared feature extractor from src.models
    - local reward/punish updates on the output layer
    - continual metrics for Task1/Task2

    It is suitable for getting the project onto a server and producing the
    first baseline runs, while keeping the code structure ready for a more
    paper-faithful SpykeTorch implementation later.
    """

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
        plasticity = self.build_plasticity(config)

        train_task1_loader = self.build_train_loader(task1.train_dataset, config)
        test_task1_loader = self.build_eval_loader(task1.test_dataset, config)
        test_task2_loader = self.build_eval_loader(task2.test_dataset, config)

        task1_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task1_loader,
            config=config,
            plasticity=plasticity,
            device=device,
            stage_name="task1",
        )
        task1_after_task1 = self.evaluate(model, test_task1_loader, device)

        self.prepare_for_task2(model=model, plasticity=plasticity, config=config)
        train_task2_loader = self.build_task2_train_loader(task1, task2, config)
        task2_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task2_loader,
            config=config,
            plasticity=plasticity,
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
            notes=self.approximation_note(),
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
        build_baseline_network = getattr(model_module, "build_baseline_network")
        model_overrides = dict(config.get("model", {}))
        return build_baseline_network(model_overrides)

    def build_plasticity(self, config: Mapping[str, Any]) -> RewardModulatedSTDP:
        train_cfg = config.get("train", {})
        plasticity_cfg = RSTDPConfig(
            learning_rate=float(train_cfg.get("learning_rate", 0.01)),
            reward_scale=float(train_cfg.get("reward_scale", 1.0)),
            punish_scale=float(train_cfg.get("punish_scale", 0.5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            freeze_fraction=float(train_cfg.get("freeze_fraction", 0.0)),
            langevin_noise_std=float(train_cfg.get("langevin_noise_std", 0.0)),
            weight_clip_min=float(train_cfg.get("weight_clip_min", -1.0)),
            weight_clip_max=float(train_cfg.get("weight_clip_max", 1.5)),
        )
        return RewardModulatedSTDP(plasticity_cfg)

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

    def prepare_for_task2(
        self,
        model: nn.Module,
        plasticity: RewardModulatedSTDP,
        config: Mapping[str, Any],
    ) -> None:
        return None

    def train_single_task(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        plasticity: RewardModulatedSTDP,
        device: torch.device,
        stage_name: str,
    ) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        epochs_key = f"num_epochs_{stage_name}"
        epochs = int(train_cfg.get(epochs_key, train_cfg.get("num_epochs", 1)))

        model.train()
        output_layer = self.get_output_layer(model)
        num_classes = int(config["model"]["num_classes"])
        neurons_per_class = int(config["model"]["neurons_per_class"])

        stage_stats = []
        for epoch_idx in range(epochs):
            epoch_samples = 0
            reward_updates = 0
            punish_updates = 0
            correct = 0

            for batch in dataloader:
                inputs, targets = move_batch_to_device(batch, device)
                features = model.forward_features(inputs)
                class_scores = features["class_scores"]
                predictions = class_scores.argmax(dim=-1)
                correct += int((predictions == targets).sum().item())
                epoch_samples += int(targets.shape[0])

                update_stats = plasticity.update_output_weights(
                    weights=output_layer.weight.data,
                    features=features["s2_flat"].detach(),
                    raw_logits=features["s3"].detach(),
                    targets=targets.detach(),
                    num_classes=num_classes,
                    neurons_per_class=neurons_per_class,
                )
                reward_updates += int(update_stats["reward_updates"])
                punish_updates += int(update_stats["punish_updates"])

            stage_stats.append(
                {
                    "epoch": epoch_idx + 1,
                    "samples": epoch_samples,
                    "train_acc_proxy": float(correct / max(epoch_samples, 1)),
                    "reward_updates": reward_updates,
                    "punish_updates": punish_updates,
                }
            )

        return {
            "stage": stage_name,
            "epochs": epochs,
            "history": stage_stats,
        }

    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataloader: Any, device: torch.device) -> float:
        model.eval()
        correct = 0
        total = 0
        for batch in dataloader:
            inputs, targets = move_batch_to_device(batch, device)
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.shape[0])
        return float(correct / max(total, 1))

    def get_output_layer(self, model: nn.Module) -> nn.Linear:
        output_layer = model.s3[1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Expected model.s3[1] to be the output Linear layer.")
        return output_layer

    def describe_plan(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "dataset": config.get("data", {}),
            "tasks": config.get("tasks", {}),
            "train": config.get("train", {}),
            "eval": config.get("eval", {}),
            "approximation": self.approximation_note(),
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

    def approximation_note(self) -> str:
        return (
            "Runnable approximation: MNIST task splits with shared feature extractor and "
            "reward-modulated local updates on the output layer. This is not yet a full "
            "paper-faithful SpykeTorch reproduction."
        )

    def _try_describe_tasks(self, config: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
        try:
            return bundle_summary(build_task_bundles(config["data"], config["tasks"]))
        except Exception:
            task_names = list(config.get("tasks", {}).get("task_names", []))
            task_splits = list(config.get("tasks", {}).get("task_splits", []))
            return [
                {"name": str(name), "labels": labels}
                for name, labels in zip(task_names, task_splits)
            ]

    def _ensure_runtime_dependencies(self) -> None:
        try:
            import_module("torchvision")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torchvision is required for the current MNIST baseline pipeline."
            ) from exc


class CatastrophicForgettingTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="catastrophic")


class JointTrainingTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="joint_training")

    def build_task2_train_loader(
        self,
        task1: TaskBundle,
        task2: TaskBundle,
        config: Mapping[str, Any],
    ) -> Any:
        joint_dataset = ConcatenatedSubset([task1.train_dataset, task2.train_dataset])
        return self.build_train_loader(joint_dataset, config)


class FrozenLargeWeightsTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="frozen_large_weights")

    def prepare_for_task2(
        self,
        model: nn.Module,
        plasticity: RewardModulatedSTDP,
        config: Mapping[str, Any],
    ) -> None:
        output_layer = self.get_output_layer(model)
        plasticity.set_frozen_mask_from_weights(output_layer.weight.data)


class LangevinTrainer(BaselineTrainer):
    def __init__(self) -> None:
        super().__init__(method_name="langevin")


TRAINER_REGISTRY = {
    "catastrophic": CatastrophicForgettingTrainer,
    "joint_training": JointTrainingTrainer,
    "frozen_large_weights": FrozenLargeWeightsTrainer,
    "langevin": LangevinTrainer,
}
