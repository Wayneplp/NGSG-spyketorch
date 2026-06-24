from __future__ import annotations

from typing import Dict


def compute_forgetting(task1_after_task1: float, task1_after_task2: float) -> float:
    return float(task1_after_task1 - task1_after_task2)


def compute_avg_acc(task1_after_task2: float, task2_after_task2: float) -> float:
    return float((task1_after_task2 + task2_after_task2) / 2.0)


def summarize_continual_metrics(
    task1_after_task1: float,
    task1_after_task2: float,
    task2_after_task2: float,
) -> Dict[str, float]:
    return {
        "task1_after_task1": float(task1_after_task1),
        "task1_after_task2": float(task1_after_task2),
        "task2_after_task2": float(task2_after_task2),
        "forgetting": compute_forgetting(task1_after_task1, task1_after_task2),
        "avg_acc": compute_avg_acc(task1_after_task2, task2_after_task2),
    }
