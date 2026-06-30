from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


@dataclass
class TaskBundle:
    name: str
    label_set: List[int]
    train_dataset: Dataset[Any]
    test_dataset: Dataset[Any]


class CachedTensorDataset(Dataset[Any]):
    """Read-only dataset backed by precomputed tensor samples on disk."""

    def __init__(self, cache_dir: Path, length: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        payload = torch.load(self.cache_dir / f"{index:06d}.pt", map_location="cpu")
        return payload["input"], int(payload["target"])


def _hash_jsonable(payload: Any) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(rendered.encode("utf-8")).hexdigest()


def _describe_dataset_for_cache(dataset: Dataset[Any]) -> Dict[str, Any]:
    if isinstance(dataset, MappedSubset):
        return {
            "type": "MappedSubset",
            "length": len(dataset),
            "indices_hash": _hash_jsonable(dataset.indices),
            "label_mapping": dict(sorted(dataset.label_mapping.items())),
            "base": _describe_dataset_for_cache(dataset.dataset),
        }

    if isinstance(dataset, Subset):
        return {
            "type": "Subset",
            "length": len(dataset),
            "indices_hash": _hash_jsonable(list(dataset.indices)),
            "base": _describe_dataset_for_cache(dataset.dataset),
        }

    description: Dict[str, Any] = {
        "type": dataset.__class__.__name__,
        "length": len(dataset),
    }
    for attr in ("root", "split", "train"):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            description[attr] = str(value) if isinstance(value, Path) else value
    return description


def build_preprocessed_tensor_cache(
    dataset: Dataset[Any],
    cache_root: Path,
    encoder: Any,
    metadata: Mapping[str, Any],
) -> CachedTensorDataset:
    """Materialize encoded tensors once, then read them back as a dataset."""

    cache_payload = {
        "version": 1,
        "metadata": dict(metadata),
        "dataset": _describe_dataset_for_cache(dataset),
    }
    fingerprint = _hash_jsonable(cache_payload)[:16]
    cache_dir = Path(cache_root) / fingerprint
    metadata_path = cache_dir / "metadata.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")

    length = len(dataset)
    missing = [idx for idx in range(length) if not (cache_dir / f"{idx:06d}.pt").exists()]
    if missing:
        print(
            f"[cache] materializing {len(missing)}/{length} samples into {cache_dir}",
            flush=True,
        )
        for position, sample_idx in enumerate(missing, start=1):
            image, target = dataset[sample_idx]
            encoded = encoder(image)
            payload = {
                "input": encoded.detach().cpu().contiguous(),
                "target": int(target),
            }
            target_path = cache_dir / f"{sample_idx:06d}.pt"
            tmp_path = cache_dir / f".{sample_idx:06d}.{os.getpid()}.tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, target_path)
            if position % 500 == 0 or position == len(missing):
                print(
                    f"[cache] saved {position}/{len(missing)} samples into {cache_dir.name}",
                    flush=True,
                )

    return CachedTensorDataset(cache_dir=cache_dir, length=length)


def _build_transform(normalize: bool) -> transforms.Compose:
    transform_steps: List[Any] = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_steps)


def _build_emnist_transform(normalize: bool, fix_orientation: bool) -> transforms.Compose:
    transform_steps: List[Any] = []
    if fix_orientation:
        transform_steps.append(lambda image: TF.hflip(TF.rotate(image, -90)))
    transform_steps.append(transforms.ToTensor())
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_steps)


def _filter_indices_by_labels(targets: Sequence[int], allowed_labels: Iterable[int]) -> List[int]:
    allowed = set(int(label) for label in allowed_labels)
    return [idx for idx, label in enumerate(targets) if int(label) in allowed]


def _balanced_indices_by_labels(
    targets: Sequence[int],
    allowed_labels: Sequence[int],
    samples_per_label: Optional[int],
    seed: int,
) -> List[int]:
    grouped: Dict[int, List[int]] = {int(label): [] for label in allowed_labels}
    allowed = set(grouped)
    for idx, label in enumerate(targets):
        label_int = int(label)
        if label_int in allowed:
            grouped[label_int].append(idx)

    rng = random.Random(seed)
    indices: List[int] = []
    for label in allowed_labels:
        label_indices = list(grouped[int(label)])
        rng.shuffle(label_indices)
        if samples_per_label is not None:
            label_indices = label_indices[: int(samples_per_label)]
        indices.extend(label_indices)
    rng.shuffle(indices)
    return indices


def _subset_by_labels(dataset: Dataset[Any], allowed_labels: Sequence[int]) -> Dataset[Any]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose a targets attribute for task splitting.")
    indices = _filter_indices_by_labels(targets, allowed_labels)
    return Subset(dataset, indices)


class MappedSubset(Dataset[Any]):
    def __init__(
        self,
        dataset: Dataset[Any],
        indices: Sequence[int],
        label_mapping: Mapping[int, int],
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.label_mapping = {int(key): int(value) for key, value in label_mapping.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        image, label = self.dataset[self.indices[index]]
        mapped_label = self.label_mapping[int(label)]
        return image, mapped_label


def _build_label_mapping(
    allowed_labels: Sequence[int],
    label_offset: int,
    mapped_labels: Optional[Sequence[int]] = None,
) -> Dict[int, int]:
    source_labels = [int(label) for label in allowed_labels]
    if mapped_labels is not None:
        target_labels = [int(label) for label in mapped_labels]
        if len(source_labels) != len(target_labels):
            raise ValueError("mapped_labels must have the same length as allowed labels.")
        return dict(zip(source_labels, target_labels))
    return {
        source_label: int(label_offset) + idx
        for idx, source_label in enumerate(source_labels)
    }


def _mapped_subset_by_labels(
    dataset: Dataset[Any],
    allowed_labels: Sequence[int],
    label_offset: int,
    mapped_labels: Optional[Sequence[int]] = None,
    samples_per_label: Optional[int] = None,
    seed: int = 0,
) -> MappedSubset:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose a targets attribute for task splitting.")
    indices = _balanced_indices_by_labels(
        targets=targets,
        allowed_labels=allowed_labels,
        samples_per_label=samples_per_label,
        seed=seed,
    )
    label_mapping = _build_label_mapping(
        allowed_labels=allowed_labels,
        label_offset=label_offset,
        mapped_labels=mapped_labels,
    )
    return MappedSubset(dataset=dataset, indices=indices, label_mapping=label_mapping)


def _build_torchvision_dataset(
    dataset_config: Mapping[str, Any],
    train: bool,
    default_data_config: Mapping[str, Any],
) -> Dataset[Any]:
    dataset_name = str(dataset_config.get("dataset_name", dataset_config.get("name", "mnist"))).lower()
    normalize = bool(dataset_config.get("normalize", default_data_config.get("normalize", False)))
    data_root = Path(str(dataset_config.get("data_root", default_data_config.get("data_root", "data"))))

    if dataset_name == "mnist":
        transform = _build_transform(normalize=normalize)
        return datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
            transform=transform,
        )

    if dataset_name == "emnist":
        transform = _build_emnist_transform(
            normalize=normalize,
            fix_orientation=bool(dataset_config.get("fix_orientation", True)),
        )
        return datasets.EMNIST(
            root=data_root,
            split=str(dataset_config.get("split", "letters")),
            train=train,
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset_name '{dataset_name}'.")


def _build_task_bundles_from_specs(
    data_config: Mapping[str, Any],
    task_specs: Sequence[Mapping[str, Any]],
) -> List[TaskBundle]:
    bundles: List[TaskBundle] = []
    base_seed = int(data_config.get("subset_seed", 0))
    for task_idx, spec in enumerate(task_specs):
        dataset_config = dict(spec.get("dataset", {}))
        labels = [int(label) for label in spec.get("labels", [])]
        if not labels:
            raise ValueError("Each task spec must define a non-empty labels list.")

        train_dataset = _build_torchvision_dataset(dataset_config, train=True, default_data_config=data_config)
        test_dataset = _build_torchvision_dataset(dataset_config, train=False, default_data_config=data_config)
        label_offset = int(spec.get("label_offset", 0))
        mapped_labels = spec.get("mapped_labels")
        train_samples_per_label = spec.get("train_samples_per_label")
        test_samples_per_label = spec.get("test_samples_per_label")
        task_seed = int(spec.get("subset_seed", base_seed + task_idx))

        bundles.append(
            TaskBundle(
                name=str(spec.get("name", f"task_{len(bundles) + 1}")),
                label_set=list(_build_label_mapping(labels, label_offset, mapped_labels).values()),
                train_dataset=_mapped_subset_by_labels(
                    train_dataset,
                    labels,
                    label_offset=label_offset,
                    mapped_labels=mapped_labels,
                    samples_per_label=None if train_samples_per_label is None else int(train_samples_per_label),
                    seed=task_seed,
                ),
                test_dataset=_mapped_subset_by_labels(
                    test_dataset,
                    labels,
                    label_offset=label_offset,
                    mapped_labels=mapped_labels,
                    samples_per_label=None if test_samples_per_label is None else int(test_samples_per_label),
                    seed=task_seed + 100_000,
                ),
            )
        )
    return bundles


def build_task_bundles(data_config: Mapping[str, Any], task_config: Mapping[str, Any]) -> List[TaskBundle]:
    task_specs = task_config.get("task_specs")
    if task_specs:
        return _build_task_bundles_from_specs(data_config, list(task_specs))

    dataset_name = str(data_config.get("dataset_name", "mnist")).lower()
    if dataset_name != "mnist":
        raise ValueError(
            f"Only 'mnist' is supported in the legacy split config, got '{dataset_name}'. "
            "Use tasks.task_specs for multi-dataset experiments."
        )

    data_root = Path(str(data_config.get("data_root", "data/mnist")))
    transform = _build_transform(normalize=bool(data_config.get("normalize", False)))

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    task_names = list(task_config.get("task_names", []))
    task_splits = list(task_config.get("task_splits", []))
    if not task_names or not task_splits or len(task_names) != len(task_splits):
        raise ValueError("tasks.task_names and tasks.task_splits must both exist and have the same length.")

    bundles: List[TaskBundle] = []
    for name, labels in zip(task_names, task_splits):
        bundles.append(
            TaskBundle(
                name=str(name),
                label_set=[int(label) for label in labels],
                train_dataset=_subset_by_labels(train_dataset, labels),
                test_dataset=_subset_by_labels(test_dataset, labels),
            )
        )
    return bundles


def build_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class ConcatenatedSubset(Dataset[Any]):
    def __init__(self, datasets_list: Sequence[Dataset[Any]]) -> None:
        self.datasets_list = list(datasets_list)
        self.cumulative_sizes: List[int] = []
        running = 0
        for dataset in self.datasets_list:
            running += len(dataset)
            self.cumulative_sizes.append(running)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_idx = 0
        while index >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        prev_cumulative = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_cumulative
        return self.datasets_list[dataset_idx][sample_idx]


def bundle_summary(bundles: Sequence[TaskBundle]) -> List[Dict[str, Any]]:
    return [
        {
            "name": bundle.name,
            "labels": bundle.label_set,
            "train_size": len(bundle.train_dataset),
            "test_size": len(bundle.test_dataset),
        }
        for bundle in bundles
    ]
