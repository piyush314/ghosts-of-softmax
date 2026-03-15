"""Optional end-to-end smoke runs for selected experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CONTRACT_PATHS = sorted(EXPERIMENTS_DIR.glob("*/contract.json"))
RUN_SMOKE = os.getenv("GHOSTS_RUN_SMOKE") == "1"


def load_contract(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


@pytest.mark.skipif(not RUN_SMOKE, reason="Set GHOSTS_RUN_SMOKE=1 to run end-to-end smoke configs.")
@pytest.mark.parametrize(
    "contract_path",
    [path for path in CONTRACT_PATHS if "smoke_run" in load_contract(path)],
    ids=lambda path: path.parent.name,
)
def test_experiment_smoke_run(contract_path: Path):
    contract = load_contract(contract_path)
    smoke = contract["smoke_run"]
    exp_dir = contract_path.parent

    proc = subprocess.run(
        [sys.executable, *smoke["command"][1:]],
        cwd=exp_dir,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    for rel_path in smoke["expected_artifacts"]:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"missing expected artifact: {path}"
