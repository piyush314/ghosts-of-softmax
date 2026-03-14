from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ghosts.reporting import json_ready, repo_relpath, scalar_stats, write_summary


def test_scalar_stats_reports_quartiles():
    stats = scalar_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["min"] == 1.0
    assert stats["median"] == 2.5
    assert stats["max"] == 4.0


def test_json_ready_handles_numpy_and_nonfinite():
    payload = {
        "value": np.float64(3.0),
        "array": np.array([1, 2]),
        "inf": float("inf"),
        "nan": float("nan"),
    }
    ready = json_ready(payload)
    assert ready["value"] == 3.0
    assert ready["array"] == [1, 2]
    assert ready["inf"] == "inf"
    assert ready["nan"] == "nan"


def test_write_summary_writes_json(tmp_path: Path):
    out = tmp_path / "summary.json"
    write_summary(out, {"metric": np.float32(1.5), "path": out})
    data = json.loads(out.read_text())
    assert data["metric"] == 1.5
    assert data["path"].endswith("summary.json")


def test_repo_relpath_returns_repo_relative(tmp_path: Path):
    repo_root = tmp_path / "repo"
    child = repo_root / "results" / "summary.json"
    child.parent.mkdir(parents=True)
    child.write_text("{}")
    assert repo_relpath(child, repo_root) == "results/summary.json"
