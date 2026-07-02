from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import hashlib
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from SpykeTorch import snn

from src.analysis.metrics import summarize_continual_metrics
from src.continual import NeuronPartition, SDPMGate
from src.plasticity import SpykeTorchRSTDPConfig, SpykeTorchRewardSTDP
from src.utils.data import (
    ConcatenatedSubset,
    TaskBundle,
    _describe_dataset_for_cache,
    _hash_jsonable,
    build_preprocessed_tensor_cache,
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
        if getattr(model, "paper_source_compatible", False):
            task1, task2 = self.prepare_paper_source_cache(task1, task2, config, model)
        rstdp = None if getattr(model, "paper_source_compatible", False) else self.build_output_rstdp(model, config).to(device)

        train_task1_loader = self.build_train_loader(task1.train_dataset, config)
        test_task1_loader = self.build_eval_loader(task1.test_dataset, config)
        test_task2_loader = self.build_eval_loader(task2.test_dataset, config)

        sdpm_gate: Optional[SDPMGate] = None
        task1_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task1_loader,
            config=config,
            rstdp=rstdp,
            device=device,
            stage_name="task1",
            sdpm_gate=sdpm_gate,
        )
        sdpm_gate = self.fit_sdpm_gate_after_task1(model, config, task1_training_stats, sdpm_gate)
        neuron_partition = self.fit_neuron_partition_after_task1(model, config, task1_training_stats)
        if neuron_partition is not None and neuron_partition.enabled:
            task1_training_stats["neuron_partition"] = neuron_partition.to_dict(include_arrays=True)

        train_task2_loader = self.build_task2_train_loader(task1, task2, config)
        if bool(config.get("train", {}).get("feature_only", False)):
            task2_training_stats = self.train_single_task(
                model=model,
                dataloader=train_task2_loader,
                config=config,
                rstdp=rstdp,
                device=device,
                stage_name="task2",
            )
            return TrainerResult(
                metrics=self.empty_metrics(),
                notes="Feature-only run: saved/reused S1/S2 checkpoints and C2 feature caches; skipped S3 training and evaluation.",
                extra={
                    "device": str(device),
                    "task_summary": bundle_summary(task_bundles),
                    "trainer_plan": self.describe_plan(config),
                    "task1_training": task1_training_stats,
                    "task2_training": task2_training_stats,
                    "model_summary": self.summarize_model(model),
                },
            )

        task1_after_task1 = self.evaluate(model, test_task1_loader, device)
        task2_training_stats = self.train_single_task(
            model=model,
            dataloader=train_task2_loader,
            config=config,
            rstdp=rstdp,
            device=device,
            stage_name="task2",
            sdpm_gate=sdpm_gate,
        )

        task1_after_task2 = self.evaluate(model, test_task1_loader, device)
        task2_after_task2 = self.evaluate(model, test_task2_loader, device)
        metrics = summarize_continual_metrics(
            task1_after_task1=task1_after_task1,
            task1_after_task2=task1_after_task2,
            task2_after_task2=task2_after_task2,
        )

        extra: Dict[str, Any] = {
            "device": str(device),
            "task_summary": bundle_summary(task_bundles),
            "trainer_plan": self.describe_plan(config),
            "task1_training": task1_training_stats,
            "task2_training": task2_training_stats,
            "model_summary": self.summarize_model(model),
        }
        if sdpm_gate is not None and sdpm_gate.enabled:
            extra["sdpm_gate"] = sdpm_gate.summarize()
        if neuron_partition is not None and neuron_partition.enabled:
            extra["neuron_partition"] = neuron_partition.summarize()

        return TrainerResult(
            metrics=metrics,
            notes=self.implementation_note(config),
            extra=extra,
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
        if architecture in {"paper_spyketorch", "paper_source", "mozafari2018"}:
            build_paper_mozafari_network = getattr(model_module, "build_paper_mozafari_network")
            return build_paper_mozafari_network(model_overrides)
        if architecture not in {"spyketorch", "official_spyketorch"}:
            raise ValueError(
                "Use architecture='paper_spyketorch' for the paper source port or architecture='spyketorch' for the older path. "
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
        prefetch_factor = train_cfg.get("prefetch_factor")
        return build_dataloader(
            dataset=dataset,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=bool(train_cfg.get("shuffle", True)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
            persistent_workers=bool(train_cfg.get("persistent_workers", False)),
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
        )

    def build_eval_loader(self, dataset: Any, config: Mapping[str, Any]) -> Any:
        eval_cfg = config.get("eval", {})
        prefetch_factor = eval_cfg.get("prefetch_factor")
        return build_dataloader(
            dataset=dataset,
            batch_size=int(eval_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", 0)),
            pin_memory=bool(eval_cfg.get("pin_memory", torch.cuda.is_available())),
            persistent_workers=bool(eval_cfg.get("persistent_workers", False)),
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
        )

    def build_task2_train_loader(
        self,
        task1: TaskBundle,
        task2: TaskBundle,
        config: Mapping[str, Any],
    ) -> Any:
        return self.build_train_loader(task2.train_dataset, config)

    def prepare_paper_source_cache(
        self,
        task1: TaskBundle,
        task2: TaskBundle,
        config: Mapping[str, Any],
        model: nn.Module,
    ) -> tuple[TaskBundle, TaskBundle]:
        data_cfg = config.get("data", {})
        if not bool(data_cfg.get("preprocess_cache", True)):
            return task1, task2

        cache_root = Path(str(data_cfg.get("preprocess_cache_root", "data/preprocessed/paper_source")))
        model_cfg = config.get("model", {})
        preprocess_meta = {
            "implementation": "paper_source_encode_v1",
            "dataset_name": data_cfg.get("dataset_name"),
            "input_size": data_cfg.get("input_size"),
            "input_channels": data_cfg.get("input_channels"),
            "time_steps": model_cfg.get("time_steps"),
            "filter_threshold": model_cfg.get("filter_threshold"),
            "local_normalization_radius": model_cfg.get("local_normalization_radius"),
        }

        def wrap_dataset(bundle: TaskBundle, split_name: str, dataset: Any) -> Any:
            return build_preprocessed_tensor_cache(
                dataset=dataset,
                cache_root=cache_root,
                encoder=model.encode,
                metadata={
                    **preprocess_meta,
                    "task_name": bundle.name,
                    "split": split_name,
                    "labels": bundle.label_set,
                },
                preload=bool(data_cfg.get("preprocess_cache_preload", False)),
                pin_memory=bool(data_cfg.get("preprocess_cache_pin_memory", False)),
            )

        return (
            TaskBundle(
                name=task1.name,
                label_set=task1.label_set,
                train_dataset=wrap_dataset(task1, "train", task1.train_dataset),
                test_dataset=wrap_dataset(task1, "test", task1.test_dataset),
            ),
            TaskBundle(
                name=task2.name,
                label_set=task2.label_set,
                train_dataset=wrap_dataset(task2, "train", task2.train_dataset),
                test_dataset=wrap_dataset(task2, "test", task2.test_dataset),
            ),
        )

    def fit_sdpm_gate_after_task1(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        task1_training_stats: Mapping[str, Any],
        sdpm_gate: Optional[SDPMGate],
    ) -> Optional[SDPMGate]:
        sdpm_cfg = config.get("continual", {}).get("sdpm_gate", {})
        if not bool(sdpm_cfg.get("enabled", False)):
            return sdpm_gate

        winner_counts = task1_training_stats.get("output_training", {}).get("winner_counts")
        if winner_counts is None:
            raise ValueError(
                "SDPM gate is enabled but Task 1 winner counts are missing. "
                "Ensure winner_frequency_log is enabled or S3 winner tracking is active."
            )

        fitted = SDPMGate.fit_from_task1_stats(
            model=model,
            winner_counts=winner_counts,
            config=sdpm_cfg,
            global_seed=int(config.get("seed", 0)),
        )
        summary = fitted.summarize()
        print(
            "[sdpm gate] fitted from Task 1 stats: "
            f"protected_fraction={summary.get('protected_fraction', 0.0):.4f} "
            f"gate_mean={summary.get('gate_mean', 0.0):.4f} "
            f"random_protection={summary.get('random_protection', False)}",
            flush=True,
        )
        return fitted

    def fit_neuron_partition_after_task1(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        task1_training_stats: Mapping[str, Any],
    ) -> Optional[NeuronPartition]:
        partition_cfg = config.get("continual", {}).get("neuron_partition", {})
        if not bool(partition_cfg.get("enabled", False)):
            return None

        output_training = task1_training_stats.get("output_training", {})
        winner_counts = output_training.get("winner_counts")
        if winner_counts is None:
            raise ValueError(
                "neuron_partition is enabled but Task 1 winner counts are missing. "
                "Ensure winner_frequency_log, sdpm_gate, or neuron_partition tracking is active."
            )

        winner_label_counts = output_training.get("winner_label_counts")
        num_classes = int(config.get("model", {}).get("num_classes", 0)) or None
        fitted = NeuronPartition.fit_from_task1_stats(
            model=model,
            winner_counts=winner_counts,
            winner_label_counts=winner_label_counts,
            config=partition_cfg,
            num_classes=num_classes,
        )
        summary = fitted.summarize()
        role_counts = summary.get("role_counts", {})
        print(
            "[neuron partition] fitted from Task 1 stats: "
            f"stable={role_counts.get('stable', 0)} "
            f"shared={role_counts.get('shared', 0)} "
            f"reserve={role_counts.get('reserve', 0)} "
            f"dead={role_counts.get('dead', 0)} "
            f"f_stable_thr={summary.get('thresholds', {}).get('f_stable_threshold', 0.0):.1f}",
            flush=True,
        )
        return fitted

    def train_single_task(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        rstdp: SpykeTorchRewardSTDP,
        device: torch.device,
        stage_name: str,
        sdpm_gate: Optional[SDPMGate] = None,
    ) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        learning_rule = str(train_cfg.get("learning_rule", "spyketorch_stdp_rstdp")).lower()
        if learning_rule not in {"spyketorch_stdp_rstdp", "spyketorch", "paper_stdp_rstdp", "paper_source_rstdp"}:
            raise ValueError(
                "Use learning_rule='spyketorch_stdp_rstdp' for the main implementation. "
                "Legacy approximation configs are stored under approx/legacy_approx/."
            )

        if getattr(model, "paper_source_compatible", False):
            return self.train_paper_single_task(
                model,
                dataloader,
                config,
                device,
                stage_name,
                sdpm_gate=sdpm_gate,
            )

        s1_epochs = self._stage_epochs(config, stage_name, "s1_stdp_epochs", 0)
        s2_epochs = self._stage_epochs(config, stage_name, "s2_stdp_epochs", 0)
        s3_epochs = int(train_cfg.get(f"num_epochs_{stage_name}", train_cfg.get("num_epochs", 1)))
        print(
            f"[stage {stage_name}] start: s1_epochs={s1_epochs}, s2_epochs={s2_epochs}, s3_epochs={s3_epochs}",
            flush=True,
        )

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

    def _paper_feature_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        return {
            "conv1": model.conv1.state_dict(),
            "conv2": model.conv2.state_dict(),
        }

    def _paper_feature_state_digest(self, model: nn.Module) -> str:
        digest = hashlib.sha1()
        for layer_name in ("conv1", "conv2"):
            state = getattr(model, layer_name).state_dict()
            for tensor_name, tensor in sorted(state.items()):
                cpu_tensor = tensor.detach().cpu().contiguous()
                digest.update(layer_name.encode("utf-8"))
                digest.update(tensor_name.encode("utf-8"))
                digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
                digest.update(str(cpu_tensor.dtype).encode("utf-8"))
                digest.update(cpu_tensor.numpy().tobytes())
        return digest.hexdigest()

    def _paper_feature_checkpoint_path(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        stage_name: str,
        s1_epochs: int,
        s2_epochs: int,
    ) -> tuple[Path, Dict[str, Any]]:
        train_cfg = config.get("train", {})
        checkpoint_cfg = train_cfg.get("feature_checkpoint", {})
        root_dir = Path(str(checkpoint_cfg.get("root_dir", "checkpoints/features")))
        dataset = getattr(dataloader, "dataset", None)
        metadata = {
            "version": 1,
            "kind": "paper_s1_s2_feature_checkpoint",
            "stage": stage_name,
            "seed": int(config.get("seed", 0)),
            "method": self.method_name,
            "model": dict(config.get("model", {})),
            "data": dict(config.get("data", {})),
            "dataset": _describe_dataset_for_cache(dataset) if dataset is not None else None,
            "s1_epochs": int(s1_epochs),
            "s2_epochs": int(s2_epochs),
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "shuffle": bool(train_cfg.get("shuffle", True)),
            "pre_state_digest": self._paper_feature_state_digest(model),
        }
        fingerprint = _hash_jsonable(metadata)[:16]
        filename = f"paper_{stage_name}_s1e{s1_epochs}_s2e{s2_epochs}_{fingerprint}.pt"
        return root_dir / filename, metadata

    def load_paper_feature_checkpoint(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        device: torch.device,
        stage_name: str,
        s1_epochs: int,
        s2_epochs: int,
    ) -> Dict[str, Any]:
        checkpoint_cfg = config.get("train", {}).get("feature_checkpoint", {})
        enabled = bool(checkpoint_cfg.get("enabled", False))
        if not enabled or not bool(checkpoint_cfg.get("load", True)) or (s1_epochs <= 0 and s2_epochs <= 0):
            return {"enabled": enabled, "loaded": False}
        checkpoint_path, metadata = self._paper_feature_checkpoint_path(model, dataloader, config, stage_name, s1_epochs, s2_epochs)
        matched_by = "exact"
        if not checkpoint_path.exists():
            fallback_path = self._find_paper_feature_checkpoint_fallback(
                expected_path=checkpoint_path,
                metadata=metadata,
                config=config,
                stage_name=stage_name,
                s1_epochs=s1_epochs,
                s2_epochs=s2_epochs,
            )
            if fallback_path is None:
                return {"enabled": True, "loaded": False, "path": str(checkpoint_path), "metadata": metadata}
            checkpoint_path = fallback_path
            matched_by = "fallback"
        payload = torch.load(checkpoint_path, map_location=device)
        model.conv1.load_state_dict(payload["conv1"])
        model.conv2.load_state_dict(payload["conv2"])
        return {
            "enabled": True,
            "loaded": True,
            "path": str(checkpoint_path),
            "matched_by": matched_by,
            "metadata": payload.get("metadata", metadata),
        }

    def _find_paper_feature_checkpoint_fallback(
        self,
        expected_path: Path,
        metadata: Mapping[str, Any],
        config: Mapping[str, Any],
        stage_name: str,
        s1_epochs: int,
        s2_epochs: int,
    ) -> Optional[Path]:
        checkpoint_cfg = config.get("train", {}).get("feature_checkpoint", {})
        if not bool(checkpoint_cfg.get("fallback_match", True)):
            return None
        root_dir = expected_path.parent
        pattern = f"paper_{stage_name}_s1e{s1_epochs}_s2e{s2_epochs}_*.pt"
        expected_model = dict(metadata.get("model", {}))
        expected_seed = int(metadata.get("seed", 0))
        candidates = sorted(root_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                payload = torch.load(candidate, map_location="cpu")
            except Exception:
                continue
            candidate_meta = payload.get("metadata", {})
            if candidate_meta.get("kind") != "paper_s1_s2_feature_checkpoint":
                continue
            if candidate_meta.get("stage") != stage_name:
                continue
            if int(candidate_meta.get("s1_epochs", -1)) != int(s1_epochs):
                continue
            if int(candidate_meta.get("s2_epochs", -1)) != int(s2_epochs):
                continue
            if int(candidate_meta.get("seed", expected_seed)) != expected_seed:
                continue
            if dict(candidate_meta.get("model", {})) != expected_model:
                continue
            return candidate
        return None

    def save_paper_feature_checkpoint(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        stage_name: str,
        s1_epochs: int,
        s2_epochs: int,
        checkpoint_info: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        checkpoint_cfg = config.get("train", {}).get("feature_checkpoint", {})
        enabled = bool(checkpoint_cfg.get("enabled", False))
        if not enabled or not bool(checkpoint_cfg.get("save", True)) or (s1_epochs <= 0 and s2_epochs <= 0):
            return {"enabled": enabled, "saved": False}
        if checkpoint_info and checkpoint_info.get("path"):
            checkpoint_path = Path(str(checkpoint_info["path"]))
            metadata = dict(checkpoint_info.get("metadata", {}))
        else:
            checkpoint_path, metadata = self._paper_feature_checkpoint_path(model, dataloader, config, stage_name, s1_epochs, s2_epochs)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "conv1": model.conv1.state_dict(),
            "conv2": model.conv2.state_dict(),
            "post_state_digest": self._paper_feature_state_digest(model),
        }
        torch.save(payload, checkpoint_path)
        print(f"[paper features] saved S1/S2 checkpoint: {checkpoint_path}", flush=True)
        return {"enabled": True, "saved": True, "path": str(checkpoint_path), "metadata": metadata}
    def build_paper_s3_input_cache_loader(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        stage_name: str,
    ) -> tuple[Any, Dict[str, Any]]:
        cache_cfg = config.get("train", {}).get("c2_feature_cache", {})
        enabled = bool(cache_cfg.get("enabled", False))
        if not enabled:
            return dataloader, {"enabled": False}
        if not hasattr(model, "extract_s3_input"):
            return dataloader, {"enabled": False, "reason": "model_has_no_extract_s3_input"}

        source_dataset = getattr(dataloader, "dataset", None)
        if source_dataset is None:
            return dataloader, {"enabled": False, "reason": "dataloader_has_no_dataset"}

        cache_root = Path(str(cache_cfg.get("root_dir", "data/features/c2")))
        metadata = {
            "implementation": "paper_c2_to_s3_input_v1",
            "stage": stage_name,
            "seed": int(config.get("seed", 0)),
            "method": self.method_name,
            "model": dict(config.get("model", {})),
            "data": dict(config.get("data", {})),
            "feature_state_digest": self._paper_feature_state_digest(model),
        }
        was_training = model.training
        model.eval()

        def encode_s3_input(sample: Any) -> torch.Tensor:
            with torch.no_grad():
                return model.extract_s3_input(sample)

        cache_preload = bool(cache_cfg.get("preload", False))
        cached_dataset = build_preprocessed_tensor_cache(
            dataset=source_dataset,
            cache_root=cache_root,
            encoder=encode_s3_input,
            metadata=metadata,
            feature_kind="paper_s3_input",
            preload=cache_preload,
            pin_memory=bool(cache_cfg.get("preload_pin_memory", False)),
        )
        if was_training:
            model.train()
        train_cfg = config.get("train", {})
        cached_batch_size = int(cache_cfg.get("batch_size", train_cfg.get("s3_cached_batch_size", train_cfg.get("batch_size", 64))))
        cached_num_workers = int(cache_cfg.get("num_workers", train_cfg.get("num_workers", 0)))
        if cache_preload and cached_num_workers > 0:
            print("[paper c2-cache] preload=true; forcing num_workers=0 to avoid duplicating the RAM cache", flush=True)
            cached_num_workers = 0
        prefetch_factor = cache_cfg.get("prefetch_factor", train_cfg.get("prefetch_factor"))
        cached_loader = build_dataloader(
            dataset=cached_dataset,
            batch_size=cached_batch_size,
            shuffle=bool(cache_cfg.get("shuffle", train_cfg.get("shuffle", True))),
            num_workers=cached_num_workers,
            pin_memory=bool(cache_cfg.get("pin_memory", train_cfg.get("pin_memory", torch.cuda.is_available()))),
            persistent_workers=bool(cache_cfg.get("persistent_workers", train_cfg.get("persistent_workers", False))),
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else None,
        )
        print(f"[paper c2-cache] using S3 input cache: {cached_dataset.cache_dir}", flush=True)
        return cached_loader, {
            "enabled": True,
            "feature_kind": "paper_s3_input",
            "cache_dir": str(cached_dataset.cache_dir),
            "batch_size": cached_batch_size,
            "num_workers": cached_num_workers,
            "pin_memory": bool(cache_cfg.get("pin_memory", train_cfg.get("pin_memory", torch.cuda.is_available()))),
            "preload": cache_preload,
            "metadata": metadata,
        }

    def _is_paper_s3_input_loader(self, dataloader: Any) -> bool:
        dataset = getattr(dataloader, "dataset", None)
        return str(getattr(dataset, "feature_kind", "")) == "paper_s3_input"
    def train_paper_single_task(
        self,
        model: nn.Module,
        dataloader: Any,
        config: Mapping[str, Any],
        device: torch.device,
        stage_name: str,
        sdpm_gate: Optional[SDPMGate] = None,
    ) -> Dict[str, Any]:
        train_cfg = config.get("train", {})
        if bool(train_cfg.get("reset_learning_rates_each_stage", True)) and hasattr(model, "reset_learning_rates"):
            model.reset_learning_rates()
        s1_epochs = self._stage_epochs(config, stage_name, "s1_stdp_epochs", 0)
        s2_epochs = self._stage_epochs(config, stage_name, "s2_stdp_epochs", 0)
        s3_epochs = int(train_cfg.get(f"num_epochs_{stage_name}", train_cfg.get("num_epochs", 1)))
        print(
            f"[paper stage {stage_name}] start: s1_epochs={s1_epochs}, s2_epochs={s2_epochs}, s3_epochs={s3_epochs}",
            flush=True,
        )
        stats: Dict[str, Any] = {
            "stage": stage_name,
            "learning_rule": "paper_source_spyketorch_stdp_anti_stdp",
            "feature_training": {},
        }

        checkpoint_info = self.load_paper_feature_checkpoint(
            model=model,
            dataloader=dataloader,
            config=config,
            device=device,
            stage_name=stage_name,
            s1_epochs=s1_epochs,
            s2_epochs=s2_epochs,
        )
        stats["feature_checkpoint"] = checkpoint_info

        if checkpoint_info.get("loaded"):
            print(f"[paper features] loaded S1/S2 checkpoint: {checkpoint_info.get('path')}", flush=True)
            stats["feature_training"]["s1"] = {"layer": "s1", "epochs": s1_epochs, "skipped": "loaded_feature_checkpoint"}
            stats["feature_training"]["s2"] = {"layer": "s2", "epochs": s2_epochs, "skipped": "loaded_feature_checkpoint"}
        else:
            if s1_epochs > 0:
                stats["feature_training"]["s1"] = self.train_paper_unsupervised(model, dataloader, device, 1, s1_epochs, train_cfg)
            if s2_epochs > 0:
                stats["feature_training"]["s2"] = self.train_paper_unsupervised(model, dataloader, device, 2, s2_epochs, train_cfg)
            stats["feature_checkpoint"] = self.save_paper_feature_checkpoint(
                model=model,
                dataloader=dataloader,
                config=config,
                stage_name=stage_name,
                s1_epochs=s1_epochs,
                s2_epochs=s2_epochs,
                checkpoint_info=checkpoint_info,
            )

        s3_dataloader, feature_cache_info = self.build_paper_s3_input_cache_loader(
            model=model,
            dataloader=dataloader,
            config=config,
            stage_name=stage_name,
        )
        stats["feature_cache"] = feature_cache_info
        if bool(train_cfg.get("feature_only", False)):
            stats["output_training"] = {"stage": "s3", "epochs": 0, "skipped": "feature_only"}
            return stats
        stats["output_training"] = self.train_paper_rstdp(
            model,
            s3_dataloader,
            device,
            s3_epochs,
            train_cfg,
            stage_name=stage_name,
            sdpm_gate=sdpm_gate,
            config=config,
        )
        return stats
    def train_paper_unsupervised(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
        layer_idx: int,
        epochs: int,
        train_cfg: Mapping[str, Any],
    ) -> Dict[str, Any]:
        progress_every = int(train_cfg.get("progress_interval_samples", 1000))
        history = []
        best_samples = 0
        stage_start = time.time()
        for epoch_idx in range(epochs):
            model.train()
            samples = 0
            epoch_start = time.time()
            for batch in dataloader:
                inputs, _ = move_batch_to_device(batch, device)
                for sample_idx in range(int(inputs.shape[0])):
                    model(inputs[sample_idx], layer_idx)
                    model.stdp(layer_idx)
                    samples += 1
                    if progress_every > 0 and samples % progress_every == 0:
                        print(f"[paper s{layer_idx}] epoch {epoch_idx + 1}/{epochs} samples={samples}", flush=True)
            best_samples = max(best_samples, samples)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - stage_start
            remaining_seconds = self._eta_seconds(elapsed_seconds, epoch_idx + 1, epochs)
            history.append({"epoch": epoch_idx + 1, "samples": samples, "stdp_updates": samples})
            print(
                f"[paper s{layer_idx}] epoch {epoch_idx + 1}/{epochs} done "
                f"samples={samples} best_samples={best_samples} "
                f"epoch_time={self._format_seconds(epoch_seconds)} "
                f"elapsed={self._format_seconds(elapsed_seconds)} "
                f"eta={self._format_seconds(remaining_seconds)}",
                flush=True,
            )
        return {"layer": f"s{layer_idx}", "epochs": epochs, "history": history}

    def train_paper_rstdp(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
        epochs: int,
        train_cfg: Mapping[str, Any],
        stage_name: str = "task1",
        sdpm_gate: Optional[SDPMGate] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        adaptive_int = float(train_cfg.get("paper_adaptive_int", 0.5))
        adaptive_min = float(train_cfg.get("paper_adaptive_min", 0.0))
        progress_every = int(train_cfg.get("progress_interval_samples", 1000))
        winner_log_cfg = train_cfg.get("winner_frequency_log", {})
        winner_log_enabled = bool(winner_log_cfg.get("enabled", False))
        sdpm_enabled = bool((config or {}).get("continual", {}).get("sdpm_gate", {}).get("enabled", False))
        partition_enabled = bool((config or {}).get("continual", {}).get("neuron_partition", {}).get("enabled", False))
        track_winner_counts = winner_log_enabled or sdpm_enabled or partition_enabled
        apply_sdpm = sdpm_gate is not None and sdpm_gate.should_apply(stage_name)
        winner_log_top_k = int(winner_log_cfg.get("top_k", 10))
        winner_log_include_counts = bool(winner_log_cfg.get("include_counts", True))
        num_s3_neurons = int(getattr(getattr(model, "config", None), "s3_neurons", len(getattr(model, "decision_map", []))))
        num_classes = int(getattr(getattr(model, "config", None), "num_classes", 0))
        apr = float(model.stdp3.learning_rate[0][0].item())
        anr = float(model.stdp3.learning_rate[0][1].item())
        app = float(model.anti_stdp3.learning_rate[0][1].item())
        anp = float(model.anti_stdp3.learning_rate[0][0].item())
        history = []
        best_acc1 = 0.0
        stage_start = time.time()
        task_winner_counts = [0] * num_s3_neurons
        task_winner_label_counts = [[0] * num_classes for _ in range(num_s3_neurons)] if num_classes > 0 else []
        using_s3_input_cache = self._is_paper_s3_input_loader(dataloader) and hasattr(model, "forward_from_s3_input")
        feature_source = "cached_c2_s3_input" if using_s3_input_cache else "raw_or_preprocessed_input"
        if apply_sdpm:
            print(f"[paper s3] SDPM gate active for stage={stage_name}", flush=True)
        for epoch_idx in range(epochs):
            model.train()
            correct = 0
            wrong = 0
            silent = 0
            samples = 0
            winner_counts = [0] * num_s3_neurons
            winner_class_counts = [0] * num_classes
            winner_label_counts = [[0] * num_classes for _ in range(num_s3_neurons)] if num_classes > 0 else []
            winner_log_samples = 0
            epoch_start = time.time()
            for batch in dataloader:
                inputs, targets = move_batch_to_device(batch, device)
                batch_correct = 0
                batch_wrong = 0
                batch_silent = 0
                batch_total = int(inputs.shape[0])
                for sample_idx in range(batch_total):
                    if using_s3_input_cache:
                        decision = int(model.forward_from_s3_input(inputs[sample_idx]))
                    else:
                        decision = int(model(inputs[sample_idx], 3))
                    target = int(targets[sample_idx].item())
                    if track_winner_counts:
                        winner_idx = self._first_winner_index(model)
                        if winner_idx is not None:
                            winner_log_samples += 1
                            if 0 <= winner_idx < len(winner_counts):
                                winner_counts[winner_idx] += 1
                            if 0 <= winner_idx < len(task_winner_counts):
                                task_winner_counts[winner_idx] += 1
                            if num_classes > 0 and 0 <= winner_idx < len(winner_label_counts) and 0 <= target < num_classes:
                                winner_label_counts[winner_idx][target] += 1
                                task_winner_label_counts[winner_idx][target] += 1
                            winner_class = self._winner_class(model, winner_idx, decision)
                            if 0 <= winner_class < len(winner_class_counts):
                                winner_class_counts[winner_class] += 1
                    if decision != -1:
                        if decision == target:
                            batch_correct += 1
                            if apply_sdpm:
                                sdpm_gate.gated_reward(model)
                            else:
                                model.reward()
                        else:
                            batch_wrong += 1
                            if apply_sdpm:
                                sdpm_gate.gated_punish(model)
                            else:
                                model.punish()
                    else:
                        batch_silent += 1
                    samples += 1
                    if progress_every > 0 and samples % progress_every == 0:
                        train_acc_proxy = float(correct + batch_correct) / max(samples, 1)
                        print(
                            f"[paper s3] epoch {epoch_idx + 1}/{epochs} samples={samples} "
                            f"train_acc_proxy={train_acc_proxy:.4f} correct={correct + batch_correct} "
                            f"wrong={wrong + batch_wrong} silent={silent + batch_silent}",
                            flush=True,
                        )
                if batch_total > 0:
                    perf_correct = batch_correct / batch_total
                    perf_wrong = batch_wrong / batch_total
                    apr_adapt = apr * (perf_wrong * adaptive_int + adaptive_min)
                    anr_adapt = anr * (perf_wrong * adaptive_int + adaptive_min)
                    app_adapt = app * (perf_correct * adaptive_int + adaptive_min)
                    anp_adapt = anp * (perf_correct * adaptive_int + adaptive_min)
                    model.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
                correct += batch_correct
                wrong += batch_wrong
                silent += batch_silent
            train_acc_proxy = float(correct / max(samples, 1))
            best_acc1 = max(best_acc1, train_acc_proxy)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - stage_start
            remaining_seconds = self._eta_seconds(elapsed_seconds, epoch_idx + 1, epochs)
            silent_rate = float(silent / max(samples, 1))
            winner_frequency = None
            winner_log_text = ""
            if winner_log_enabled:
                winner_frequency = self._summarize_winner_frequency(
                    model=model,
                    winner_counts=winner_counts,
                    winner_class_counts=winner_class_counts,
                    total_winners=winner_log_samples,
                    top_k=winner_log_top_k,
                    include_counts=winner_log_include_counts,
                )
                winner_log_text = (
                    f" winner_active={winner_frequency['active_neurons']}/{winner_frequency['total_neurons']}"
                    f" max_winner_fraction={winner_frequency['max_winner_fraction']:.4f}"
                )
            epoch_record = {
                "epoch": epoch_idx + 1,
                "samples": samples,
                "train_acc_proxy": train_acc_proxy,
                "correct": correct,
                "wrong": wrong,
                "silent": silent,
            }
            if winner_frequency is not None:
                epoch_record["winner_frequency"] = winner_frequency
            history.append(epoch_record)
            print(
                f"[paper s3] epoch {epoch_idx + 1}/{epochs} done "
                f"samples={samples} acc1={train_acc_proxy:.4f} best_acc1={best_acc1:.4f} "
                f"correct={correct} wrong={wrong} silent={silent} silent_rate={silent_rate:.4f}"
                f"{winner_log_text} "
                f"epoch_time={self._format_seconds(epoch_seconds)} "
                f"elapsed={self._format_seconds(elapsed_seconds)} "
                f"eta={self._format_seconds(remaining_seconds)}",
                flush=True,
            )
        output_stats: Dict[str, Any] = {
            "stage": "s3",
            "epochs": epochs,
            "learning_rule": "paper_source_rstdp",
            "feature_source": feature_source,
            "history": history,
            "winner_counts": task_winner_counts,
        }
        if track_winner_counts and task_winner_label_counts:
            output_stats["winner_label_counts"] = task_winner_label_counts
        if apply_sdpm and sdpm_gate is not None:
            output_stats["sdpm_gate"] = sdpm_gate.summarize()
        return output_stats

    def _first_winner_index(self, model: nn.Module) -> Optional[int]:
        winners = getattr(model, "ctx", {}).get("winners") if hasattr(model, "ctx") else None
        if winners is None or len(winners) == 0:
            return None
        try:
            return int(winners[0][0])
        except (TypeError, ValueError, IndexError):
            return None

    def _winner_class(self, model: nn.Module, winner_idx: int, fallback_decision: int) -> int:
        decision_map = getattr(model, "decision_map", None)
        if decision_map is not None and 0 <= winner_idx < len(decision_map):
            return int(decision_map[winner_idx])
        return int(fallback_decision)

    def _summarize_winner_frequency(
        self,
        model: nn.Module,
        winner_counts: Sequence[int],
        winner_class_counts: Sequence[int],
        total_winners: int,
        top_k: int,
        include_counts: bool,
    ) -> Dict[str, Any]:
        counts = [int(count) for count in winner_counts]
        total_neurons = len(counts)
        active_neurons = sum(1 for count in counts if count > 0)
        max_count = max(counts) if counts else 0
        top = sorted(enumerate(counts), key=lambda item: (-item[1], item[0]))[: max(int(top_k), 0)]
        top_winners = [
            {
                "neuron": int(neuron_idx),
                "class": self._winner_class(model, int(neuron_idx), -1),
                "count": int(count),
                "fraction": float(count / max(total_winners, 1)),
            }
            for neuron_idx, count in top
            if count > 0
        ]
        per_class_active = [0] * len(winner_class_counts)
        for neuron_idx, count in enumerate(counts):
            if count <= 0:
                continue
            winner_class = self._winner_class(model, neuron_idx, -1)
            if 0 <= winner_class < len(per_class_active):
                per_class_active[winner_class] += 1
        summary: Dict[str, Any] = {
            "total_winners": int(total_winners),
            "total_neurons": int(total_neurons),
            "active_neurons": int(active_neurons),
            "dead_neurons": int(total_neurons - active_neurons),
            "max_winner_count": int(max_count),
            "max_winner_fraction": float(max_count / max(total_winners, 1)),
            "per_class_wins": [int(count) for count in winner_class_counts],
            "per_class_active_neurons": [int(count) for count in per_class_active],
            "top_winners": top_winners,
        }
        if include_counts:
            summary["winner_counts"] = counts
        return summary

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
        update_idx = 0
        progress_every = int(train_cfg.get("progress_interval_samples", 1000))
        best_samples = 0
        stage_start = time.time()
        for epoch_idx in range(epochs):
            samples = 0
            epoch_start = time.time()
            for image, _ in self.iter_samples(dataloader, device):
                encoded = model.encode(image)
                s1 = model.s1_step(encoded)
                self._apply_stdp_schedule(stdp, train_cfg, update_idx)
                stdp(encoded, s1["potentials"], s1["spikes"], kwta=kwta, inhibition_radius=radius)
                samples += 1
                update_idx += 1
                if progress_every > 0 and samples % progress_every == 0:
                    print(f"[s1] epoch {epoch_idx + 1}/{epochs} samples={samples}", flush=True)
            best_samples = max(best_samples, samples)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - stage_start
            remaining_seconds = self._eta_seconds(elapsed_seconds, epoch_idx + 1, epochs)
            history.append({"epoch": epoch_idx + 1, "samples": samples, "stdp_updates": samples})
            print(
                f"[s1] epoch {epoch_idx + 1}/{epochs} done "
                f"samples={samples} best_samples={best_samples} "
                f"epoch_time={self._format_seconds(epoch_seconds)} "
                f"elapsed={self._format_seconds(elapsed_seconds)} "
                f"eta={self._format_seconds(remaining_seconds)}",
                flush=True,
            )
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
        update_idx = 0
        progress_every = int(train_cfg.get("progress_interval_samples", 1000))
        best_samples = 0
        stage_start = time.time()
        for epoch_idx in range(epochs):
            samples = 0
            epoch_start = time.time()
            for image, _ in self.iter_samples(dataloader, device):
                encoded = model.encode(image)
                s1 = model.s1_step(encoded)
                s2 = model.s2_step(s1["pooled"])
                self._apply_stdp_schedule(stdp, train_cfg, update_idx)
                stdp(s1["pooled"], s2["potentials"], s2["spikes"], kwta=kwta, inhibition_radius=radius)
                samples += 1
                update_idx += 1
                if progress_every > 0 and samples % progress_every == 0:
                    print(f"[s2] epoch {epoch_idx + 1}/{epochs} samples={samples}", flush=True)
            best_samples = max(best_samples, samples)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - stage_start
            remaining_seconds = self._eta_seconds(elapsed_seconds, epoch_idx + 1, epochs)
            history.append({"epoch": epoch_idx + 1, "samples": samples, "stdp_updates": samples})
            print(
                f"[s2] epoch {epoch_idx + 1}/{epochs} done "
                f"samples={samples} best_samples={best_samples} "
                f"epoch_time={self._format_seconds(epoch_seconds)} "
                f"elapsed={self._format_seconds(elapsed_seconds)} "
                f"eta={self._format_seconds(remaining_seconds)}",
                flush=True,
            )
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
        update_idx = 0
        progress_every = int(train_cfg.get("progress_interval_samples", 1000))
        best_acc1 = 0.0
        stage_start = time.time()
        for epoch_idx in range(epochs):
            samples = 0
            correct = 0
            reward_updates = 0
            punish_updates = 0
            epoch_start = time.time()
            for image, target in self.iter_samples(dataloader, device):
                features = model.forward_spikes(image)
                self._apply_rstdp_schedule(rstdp, train_cfg, update_idx)
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
                update_idx += 1
                correct += int(update_stats["prediction"] == int(target.item()))
                reward_updates += int(update_stats["reward_updates"])
                punish_updates += int(update_stats["punish_updates"])
                if progress_every > 0 and samples % progress_every == 0:
                    train_acc_proxy = float(correct / max(samples, 1))
                    print(
                        f"[s3] epoch {epoch_idx + 1}/{epochs} samples={samples} "
                        f"train_acc_proxy={train_acc_proxy:.4f} reward={reward_updates} punish={punish_updates}",
                        flush=True,
                    )
            train_acc_proxy = float(correct / max(samples, 1))
            best_acc1 = max(best_acc1, train_acc_proxy)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - stage_start
            remaining_seconds = self._eta_seconds(elapsed_seconds, epoch_idx + 1, epochs)
            history.append(
                {
                    "epoch": epoch_idx + 1,
                    "samples": samples,
                    "train_acc_proxy": train_acc_proxy,
                    "reward_updates": reward_updates,
                    "punish_updates": punish_updates,
                }
            )
            print(
                f"[s3] epoch {epoch_idx + 1}/{epochs} done "
                f"samples={samples} acc1={train_acc_proxy:.4f} best_acc1={best_acc1:.4f} "
                f"reward={reward_updates} punish={punish_updates} "
                f"epoch_time={self._format_seconds(epoch_seconds)} "
                f"elapsed={self._format_seconds(elapsed_seconds)} "
                f"eta={self._format_seconds(remaining_seconds)}",
                flush=True,
            )
        return {"stage": "s3", "epochs": epochs, "learning_rule": "spyketorch_stdp_anti_stdp", "history": history}

    def _eta_seconds(self, elapsed_seconds: float, completed_epochs: int, total_epochs: int) -> float:
        if completed_epochs <= 0 or total_epochs <= completed_epochs:
            return 0.0
        avg_epoch_seconds = float(elapsed_seconds) / float(completed_epochs)
        return avg_epoch_seconds * float(total_epochs - completed_epochs)

    def _format_seconds(self, seconds: float) -> str:
        total = max(int(round(seconds)), 0)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _scheduled_rate(self, base_rate: float, train_cfg: Mapping[str, Any], update_idx: int) -> float:
        every = int(train_cfg.get("stdp_lr_multiply_every", 0))
        if every <= 0:
            return float(base_rate)
        factor = float(train_cfg.get("stdp_lr_multiply_factor", 1.0))
        scaled = float(base_rate) * (factor ** (int(update_idx) // every))
        if scaled >= 0:
            return min(scaled, float(train_cfg.get("stdp_max_a_plus", scaled)))
        return max(scaled, float(train_cfg.get("stdp_min_a_minus", scaled)))

    def _apply_stdp_schedule(self, stdp: snn.STDP, train_cfg: Mapping[str, Any], update_idx: int) -> None:
        stdp.update_all_learning_rate(
            self._scheduled_rate(float(train_cfg.get("stdp_a_plus", 0.004)), train_cfg, update_idx),
            self._scheduled_rate(float(train_cfg.get("stdp_a_minus", -0.003)), train_cfg, update_idx),
        )

    def _apply_rstdp_schedule(self, rstdp: SpykeTorchRewardSTDP, train_cfg: Mapping[str, Any], update_idx: int) -> None:
        rstdp.update_learning_rate(
            self._scheduled_rate(float(train_cfg.get("reward_active", 0.004)), train_cfg, update_idx),
            self._scheduled_rate(float(train_cfg.get("reward_inactive", -0.003)), train_cfg, update_idx),
            self._scheduled_rate(float(train_cfg.get("punish_active", -0.003)), train_cfg, update_idx),
            self._scheduled_rate(float(train_cfg.get("punish_inactive", 0.0005)), train_cfg, update_idx),
        )

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
        model_cfg = config.get("model", {})
        architecture = str(model_cfg.get("architecture", "spyketorch")).lower()
        implementation = (
            "paper-source port of dmitryanton68/continuous_learning MozafariMNIST2018"
            if architecture in {"paper_spyketorch", "paper_source", "mozafari2018"}
            else "official SpykeTorch package tutorial-style path"
        )
        return {
            "method": self.method_name,
            "dataset": config.get("data", {}),
            "tasks": config.get("tasks", {}),
            "train": config.get("train", {}),
            "eval": config.get("eval", {}),
            "continual": config.get("continual", {}),
            "implementation": implementation,
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
        architecture = str((config or {}).get("model", {}).get("architecture", "spyketorch")).lower()
        sdpm_enabled = bool((config or {}).get("continual", {}).get("sdpm_gate", {}).get("enabled", False))
        partition_enabled = bool((config or {}).get("continual", {}).get("neuron_partition", {}).get("enabled", False))
        if architecture in {"paper_spyketorch", "paper_source", "mozafari2018"}:
            note = (
                "Paper-source port: model/preprocessing/forward/STDP/anti-STDP follow "
                "dmitryanton68/continuous_learning MozafariMNIST2018 notebooks, backed by "
                "the official SpykeTorch package."
            )
            if sdpm_enabled:
                note += " SDPM gate scales S3 reward/anti-STDP updates using Task 1 winner-frequency and weight-strength importance."
            if partition_enabled:
                note += " Neuron partition assigns S3 neurons to stable/shared/reserve pools from Task 1 winner statistics."
            return note
        return (
            "Official SpykeTorch-based tutorial path: S1/S2 use SpykeTorch snn.STDP, "
            "S1/S2/S3 layers are SpykeTorch modules, and S3 uses snn.STDP plus anti-STDP."
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
