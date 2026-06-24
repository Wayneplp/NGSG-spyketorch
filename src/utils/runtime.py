from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_batch_to_device(batch: Iterable[Any], device: torch.device) -> tuple[Any, ...]:
    moved = []
    for item in batch:
        if hasattr(item, "to"):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def stringify_scalar_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    rendered: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            rendered[key] = str(value)
        else:
            rendered[key] = value
    return rendered
