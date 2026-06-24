from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class TaskBundle:
    name: str
    label_set: List[int]
    train_dataset: Dataset[Any]
    test_dataset: Dataset[Any]


def _build_transform(normalize: bool) -> transforms.Compose:
    transform_steps: List[Any] = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_steps)


def _filter_indices_by_labels(targets: Sequence[int], allowed_labels: Iterable[int]) -> List[int]:
    allowed = set(int(label) for label in allowed_labels)
    return [idx for idx, label in enumerate(targets) if int(label) in allowed]


def _subset_by_labels(dataset: Dataset[Any], allowed_labels: Sequence[int]) -> Dataset[Any]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose a targets attribute for task splitting.")
    indices = _filter_indices_by_labels(targets, allowed_labels)
    return Subset(dataset, indices)


def build_task_bundles(data_config: Mapping[str, Any], task_config: Mapping[str, Any]) -> List[TaskBundle]:
    dataset_name = str(data_config.get("dataset_name", "mnist")).lower()
    if dataset_name != "mnist":
        raise ValueError(
            f"Only 'mnist' is supported in the current runnable baseline, got '{dataset_name}'."
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
