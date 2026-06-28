from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from src.analysis.metrics import summarize_continual_metrics
from src.plasticity.rstdp import RSTDPConfig, RewardModulatedSTDP
from src.plasticity.stdp import LocalConvSTDP, LocalSTDPConfig
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
    """Continual-learning baseline trainer for the reproduction workspace."""

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
            notes=self.approximation_note(config),
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

        if architecture in {"supervised_mlp", "mlp"}:
            build_supervised_mnist_network = getattr(
                model_module,
                "build_supervised_mnist_network",
            )
            return build_supervised_mnist_network(model_overrides)

        build_baseline_network = getattr(model_module, "build_baseline_network")
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
            weight_clip_min=float(train_cfg.get("weight_clip_min", 0.0)),
            weight_clip_max=float(train_cfg.get("weight_clip_max", 1.0)),
            active_threshold=float(train_cfg.get("rstdp_active_threshold", 0.5)),
            reward_active=float(train_cfg.get("reward_active", 0.004)),
            reward_inactive=float(train_cfg.get("reward_inactive", -0.003)),
            punish_active=float(train_cfg.get("punish_active", -0.003)),
            punish_inactive=float(train_cfg.get("punish_inactive", 0.0005)),
            use_weight_stabilizer=bool(train_cfg.get("use_weight_stabilizer", True)),
        )
        return RewardModulatedSTDP(plasticity_cfg)

    def build_local_stdp(self, config: Mapping[str, Any]) -> LocalConvSTDP:
        train_cfg = config.get("train", {})
        return LocalConvSTDP(
            LocalSTDPConfig(
                a_plus=float(train_cfg.get("stdp_a_plus", 0.004)),
                a_minus=float(train_cfg.get("stdp_a_minus", -0.003)),
                lr_multiply_every=int(train_cfg.get("stdp_lr_multiply_every", 500)),
                lr_multiply_factor=float(train_cfg.get("stdp_lr_multiply_factor", 2.0)),
                max_a_plus=float(train_cfg.get("stdp_max_a_plus", 0.15)),
                min_a_minus=float(train_cfg.get("stdp_min_a_minus", -0.1125)),
                weight_min=float(train_cfg.get("weight_clip_min", 0.0)),
                weight_max=float(train_cfg.get("weight_clip_max", 1.0)),
                active_threshold=float(train_cfg.get("stdp_active_threshold", 0.5)),
                winners_per_sample=int(train_cfg.get("stdp_winners_per_sample", 1)),
                max_updates_per_batch=int(train_cfg.get("stdp_max_updates_per_batch", 256)),
            )
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
        learning_rule = str(train_cfg.get("learning_rule", "rstdp")).lower()

        if learning_rule in {"backprop", "supervised"}:
            return self.train_single_task_backprop(
                model=model,
                dataloader=dataloader,
                config=config,
                device=device,
                stage_name=stage_name,
                epochs=epochs,
            )

        if learning_rule in {"paper_stdp_rstdp", "stdp_rstdp"}:
            return self.train_single_task_paper_local(
                model=model,
                dataloader=dataloader,
                config=config,
                plasticity=plasticity,
                device=device,
                stage_name=stage_name,
                output_epochs=epochs,
            )

        return self.train_single_task_rstdp_output(
            model=model,
            dataloader=dataloader,
            config=config,
            plasticity=plasticity,
            device=device,
            stage_name=stage_name,
            epochs=epochs,
        )

    def train_single_task_paper_local(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        plasticity: RewardModulatedSTDP,
        device: torch.device,
        stage_name: str,
        output_epochs: int,
    ) -> Dict[str, Any]:
        s1_epochs = self._stage_epochs(config, stage_name, "s1_stdp_epochs", 0)
        s2_epochs = self._stage_epochs(config, stage_name, "s2_stdp_epochs", 0)
        stats: Dict[str, Any] = {
            "stage": stage_name,
            "learning_rule": "paper_stdp_rstdp",
            "feature_training": {},
        }

        if s1_epochs > 0:
            stats["feature_training"]["s1"] = self.train_conv_stdp_layer(
                model=model,
                dataloader=dataloader,
                config=config,
                device=device,
                layer_name="s1",
                epochs=s1_epochs,
            )
        if s2_epochs > 0:
            stats["feature_training"]["s2"] = self.train_conv_stdp_layer(
                model=model,
                dataloader=dataloader,
                config=config,
                device=device,
                layer_name="s2",
                epochs=s2_epochs,
            )

        stats["output_training"] = self.train_single_task_rstdp_output(
            model=model,
            dataloader=dataloader,
            config=config,
            plasticity=plasticity,
            device=device,
            stage_name=stage_name,
            epochs=output_epochs,
        )
        return stats

    def train_conv_stdp_layer(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        device: torch.device,
        layer_name: str,
        epochs: int,
    ) -> Dict[str, Any]:
        updater = self.build_local_stdp(config)
        history = []
        model.train()
        for epoch_idx in range(epochs):
            samples = 0
            updates = 0
            last_update: Dict[str, Any] = {}
            for batch in dataloader:
                inputs, _ = move_batch_to_device(batch, device)
                layer_inputs = self._layer_stdp_inputs(model, inputs, layer_name)
                conv = self._layer_conv(model, layer_name)
                update_stats = updater.update_conv_weights(conv, layer_inputs.detach())
                samples += int(inputs.shape[0])
                updates += int(update_stats["updates"])
                last_update = update_stats
            history.append(
                {
                    "epoch": epoch_idx + 1,
                    "samples": samples,
                    "stdp_updates": updates,
                    "a_plus": float(last_update.get("a_plus", updater.current_a_plus)),
                    "a_minus": float(last_update.get("a_minus", updater.current_a_minus)),
                }
            )
        return {
            "layer": layer_name,
            "epochs": epochs,
            "history": history,
        }

    def train_single_task_rstdp_output(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        plasticity: RewardModulatedSTDP,
        device: torch.device,
        stage_name: str,
        epochs: int,
    ) -> Dict[str, Any]:
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
            "learning_rule": "rstdp_output",
            "history": stage_stats,
        }

    def train_single_task_backprop(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        device: torch.device,
        stage_name: str,
        epochs: int,
    ) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        learning_rate = float(train_cfg.get("learning_rate", 0.01))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        momentum = float(train_cfg.get("momentum", 0.9))
        optimizer_name = str(train_cfg.get("optimizer", "sgd")).lower()

        parameters = [param for param in model.parameters() if param.requires_grad]
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        criterion = nn.CrossEntropyLoss()
        stage_stats = []
        for epoch_idx in range(epochs):
            model.train()
            epoch_samples = 0
            correct = 0
            total_loss = 0.0

            for batch in dataloader:
                inputs, targets = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                predictions = logits.argmax(dim=-1)
                batch_size = int(targets.shape[0])
                correct += int((predictions == targets).sum().item())
                epoch_samples += batch_size
                total_loss += float(loss.item()) * batch_size

            stage_stats.append(
                {
                    "epoch": epoch_idx + 1,
                    "samples": epoch_samples,
                    "train_acc": float(correct / max(epoch_samples, 1)),
                    "train_loss": float(total_loss / max(epoch_samples, 1)),
                    "optimizer": optimizer_name,
                    "optimizer_updates": len(dataloader),
                }
            )

        return {
            "stage": stage_name,
            "epochs": epochs,
            "learning_rule": "backprop",
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

    def _stage_epochs(
        self,
        config: Mapping[str, Any],
        stage_name: str,
        key: str,
        default: int,
    ) -> int:
        train_cfg = config.get("train", {})
        stage_key = f"{key}_{stage_name}"
        return int(train_cfg.get(stage_key, train_cfg.get(key, default)))

    @torch.no_grad()
    def _layer_stdp_inputs(self, model: nn.Module, inputs: torch.Tensor, layer_name: str) -> torch.Tensor:
        dog = model.preprocessor(inputs)
        latency = model.encoder(dog)
        if layer_name == "s1":
            return latency
        if layer_name == "s2":
            return model.s1(latency)
        raise ValueError(f"Unsupported STDP layer '{layer_name}'.")

    def _layer_conv(self, model: nn.Module, layer_name: str) -> nn.Conv2d:
        layer = getattr(model, layer_name)
        conv = getattr(layer, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"Expected model.{layer_name}.conv to be nn.Conv2d.")
        return conv

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
            "approximation": self.approximation_note(config),
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

    def approximation_note(self, config: Optional[Mapping[str, Any]] = None) -> str:
        learning_rule = str((config or {}).get("train", {}).get("learning_rule", "rstdp")).lower()
        if learning_rule in {"backprop", "supervised"}:
            return (
                "Supervised continual-learning sanity baseline: train Task 1, then train "
                "Task 2 on the same network without replay or protection."
            )
        if learning_rule in {"paper_stdp_rstdp", "stdp_rstdp"}:
            return (
                "Paper-protocol local-learning baseline: S1/S2 use layer-wise local STDP "
                "and S3 uses reward-modulated STDP. This is an in-repository approximation "
                "of the paper's SpykeTorch dynamics, not the original SpykeTorch package."
            )
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
                "torchvision is required for the current MNIST/EMNIST baseline pipeline."
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