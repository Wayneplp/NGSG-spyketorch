"""Single role-path selector: always compare stable vs plastic subset WTA."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROLE_PATH_FEATURE_NAMES = (
    "stable_top1",
    "stable_top2",
    "stable_margin",
    "plastic_top1",
    "plastic_top2",
    "plastic_margin",
    "stable_minus_plastic_top1",
)


def _top2_from_scores(scores: Sequence[float], allow_indices: Sequence[int]) -> Tuple[float, float, float]:
    values = [float(scores[i]) for i in allow_indices if 0 <= int(i) < len(scores) and math.isfinite(float(scores[i]))]
    if not values:
        return float("-inf"), float("-inf"), float("nan")
    ordered = sorted(values, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else ordered[0]
    return top1, top2, top1 - top2


def extract_role_path_features(
    neuron_scores: Sequence[float],
    *,
    stable_indices: Sequence[int],
    plastic_indices: Sequence[int],
) -> List[float]:
    vs1, vs2, ds = _top2_from_scores(neuron_scores, stable_indices)
    vp1, vp2, dp = _top2_from_scores(neuron_scores, plastic_indices)
    values = [vs1, vs2, ds, vp1, vp2, dp, vs1 - vp1]
    if len(values) != len(ROLE_PATH_FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("Role-path features must be finite and 7-dimensional.")
    return values


def role_path_label(stable_correct: bool, plastic_correct: bool) -> Optional[int]:
    """Return 0/1 when paths disagree; None when both agree (correct or wrong)."""
    if stable_correct and not plastic_correct:
        return 0
    if plastic_correct and not stable_correct:
        return 1
    return None


def sample_weight(label: Optional[int], *, disagree_weight: float = 1.0, harmful_weight: float = 3.0, neutral_weight: float = 0.0) -> float:
    """Weight for training. Neutral (agree) samples default to weight 0 (ignored)."""
    if label is None:
        return float(neutral_weight)
    # Both disagree classes use disagree_weight; callers can up-weight harmful via oversampling.
    return float(disagree_weight)


def stratified_disagree_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    fit_fraction: float = 0.7,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    by_label: Dict[int, List[Mapping[str, Any]]] = {0: [], 1: []}
    for row in rows:
        label = role_path_label(bool(row["stable_correct"]), bool(row["plastic_correct"]))
        if label is None:
            continue
        by_label[int(label)].append(row)
    if any(len(bucket) < 2 for bucket in by_label.values()):
        raise ValueError("Need at least two disagree samples for each path preference.")
    rng = random.Random(int(seed))
    fit: List[Mapping[str, Any]] = []
    holdout: List[Mapping[str, Any]] = []
    for bucket in by_label.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        cut = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * float(fit_fraction)))))
        fit.extend(shuffled[:cut])
        holdout.extend(shuffled[cut:])
    rng.shuffle(fit)
    rng.shuffle(holdout)
    return fit, holdout


class RolePathSelectorMLP:
    """Tiny MLP that picks plastic path when sigmoid(logit) >= threshold."""

    def __init__(self, input_dim: int = 7, *, seed: int = 0) -> None:
        import torch

        torch.manual_seed(int(seed))
        self.mean = None
        self.std = None
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        weights: Sequence[float],
        *,
        epochs: int = 400,
        harmful_weight: float = 3.0,
    ) -> None:
        import torch

        if not features or len(features) != len(targets) or len(features) != len(weights):
            raise ValueError("Role-path fit arrays must be non-empty and aligned.")
        # Up-weight choosing plastic when target=1 is wrong historically is handled by
        # harmful_weight on target=0 (preferring stable wrongly is less common for Task2);
        # apply harmful_weight when target==0 to penalize predicting plastic on stable-correct.
        adjusted = []
        for target, weight in zip(targets, weights):
            w = float(weight)
            if int(target) == 0:
                w *= float(harmful_weight)
            adjusted.append(w)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        w = torch.tensor(adjusted, dtype=torch.float32).reshape(-1, 1)
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        x = (x - self.mean) / self.std
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=2e-3, weight_decay=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.network.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            loss = (loss_fn(self.network(x), y) * w).sum() / w.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()

    def predict_plastic_probability(self, features: Sequence[Sequence[float]]) -> List[float]:
        import torch

        if not features:
            return []
        if self.mean is None or self.std is None:
            raise RuntimeError("RolePathSelectorMLP must be fitted before prediction.")
        x = torch.tensor(features, dtype=torch.float32)
        x = (x - self.mean) / self.std
        self.network.eval()
        with torch.no_grad():
            return torch.sigmoid(self.network(x)).reshape(-1).tolist()

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.network.parameters()))


def choose_probability_threshold(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
) -> Dict[str, Any]:
    """Pick threshold maximizing correct-path choices on disagree calibration rows."""
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must align.")
    candidates = sorted({0.0, 1.0, *(float(p) for p in probabilities)})
    best = None
    for threshold in candidates:
        correct = 0
        total = 0
        for row, prob in zip(rows, probabilities):
            label = role_path_label(bool(row["stable_correct"]), bool(row["plastic_correct"]))
            if label is None:
                continue
            pred = 1 if float(prob) >= threshold else 0
            correct += int(pred == label)
            total += 1
        key = (correct / max(total, 1), -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold), {"correct": correct, "total": total, "accuracy": correct / max(total, 1)})
    assert best is not None
    return {"threshold": best[1], "summary": best[2]}


def apply_role_path_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    probabilities: Sequence[float],
    threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must align.")
    by_task = {0: [0, 0], 1: [0, 0]}
    annotated: List[Dict[str, Any]] = []
    pick_plastic = disagree = disagree_correct = 0
    for row, prob in zip(rows, probabilities):
        use_plastic = float(prob) >= float(threshold)
        pred = int(row["plastic_prediction"] if use_plastic else row["stable_prediction"])
        true_y = int(row["true_class"])
        task = int(row["true_task"])
        by_task[task][1] += 1
        by_task[task][0] += int(pred == true_y)
        pick_plastic += int(use_plastic)
        label = role_path_label(bool(row["stable_correct"]), bool(row["plastic_correct"]))
        if label is not None:
            disagree += 1
            disagree_correct += int((1 if use_plastic else 0) == label)
        annotated.append(
            {
                **dict(row),
                "selector_plastic_prob": float(prob),
                "selector_pick_plastic": bool(use_plastic),
                "selector_prediction": pred,
                "selector_correct": bool(pred == true_y),
            }
        )
    t1 = by_task[0][0] / max(by_task[0][1], 1)
    t2 = by_task[1][0] / max(by_task[1][1], 1)
    return {
        "task1_accuracy": float(t1),
        "task2_accuracy": float(t2),
        "macro_average": float((t1 + t2) / 2.0),
        "pick_plastic_rate": float(pick_plastic / max(len(rows), 1)),
        "disagree_count": int(disagree),
        "disagree_path_accuracy": float(disagree_correct / max(disagree, 1)),
        "n": int(len(rows)),
    }, annotated


def evaluate_fixed_path(rows: Sequence[Mapping[str, Any]], which: str) -> Dict[str, Any]:
    key = "stable_prediction" if which == "stable" else "plastic_prediction" if which == "plastic" else None
    if which == "natural":
        key = "natural_prediction"
    elif which == "oracle":
        key = "oracle_prediction"
    elif which == "margin":
        key = None
    if which not in {"stable", "plastic", "natural", "oracle", "margin"}:
        raise ValueError(f"Unknown path mode: {which}")
    by_task = {0: [0, 0], 1: [0, 0]}
    for row in rows:
        task = int(row["true_task"])
        true_y = int(row["true_class"])
        if which == "margin":
            # Prefer higher role-subset top1 membrane score.
            pred = int(
                row["plastic_prediction"]
                if float(row["features"][3]) > float(row["features"][0])
                else row["stable_prediction"]
            )
        else:
            pred = int(row[key])  # type: ignore[index]
        by_task[task][1] += 1
        by_task[task][0] += int(pred == true_y)
    t1 = by_task[0][0] / max(by_task[0][1], 1)
    t2 = by_task[1][0] / max(by_task[1][1], 1)
    return {
        "task1_accuracy": float(t1),
        "task2_accuracy": float(t2),
        "macro_average": float((t1 + t2) / 2.0),
        "n": int(len(rows)),
    }
