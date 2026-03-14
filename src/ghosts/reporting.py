"""Helpers for machine-readable experiment summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def scalar_stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": int(arr.size),
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(finite)),
        "q25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "q75": float(np.percentile(finite, 75)),
        "max": float(np.max(finite)),
    }


def repo_relpath(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
