"""Smoke tests and schema checks for experiment reproducibility contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CONTRACT_PATHS = sorted(EXPERIMENTS_DIR.glob("*/contract.json"))


def load_contract(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def test_contract_count_and_basic_files():
    # Keep this as a lower bound so new experiments can be added without
    # rewriting the test, while still catching accidental contract loss.
    assert len(CONTRACT_PATHS) >= 8
    for contract_path in CONTRACT_PATHS:
        exp_dir = contract_path.parent
        assert (exp_dir / "README.md").exists()
        assert (exp_dir / "run.py").exists()


def test_contract_schema_and_paths():
    required = {
        "contract_version",
        "id",
        "title",
        "paper_result",
        "dataset",
        "entry_point",
        "smoke_test",
        "canonical_command",
        "artifacts",
        "notes",
    }
    for contract_path in CONTRACT_PATHS:
        contract = load_contract(contract_path)
        assert required.issubset(contract.keys())
        assert contract["contract_version"] == 1
        assert isinstance(contract["canonical_command"], list)
        assert contract["canonical_command"][:2] == ["python", contract["entry_point"]]
        assert isinstance(contract["artifacts"], list)
        assert contract["artifacts"]
        assert (contract_path.parent / contract["entry_point"]).exists()
        assert any(rel_path.endswith("summary.json") for rel_path in contract["artifacts"])
        if "smoke_run" in contract:
            smoke_run = contract["smoke_run"]
            assert smoke_run["kind"] == "command"
            assert smoke_run["command"][:2] == ["python", contract["entry_point"]]
            assert smoke_run["expected_artifacts"]
        for rel_path in contract["artifacts"]:
            assert not Path(rel_path).is_absolute()


def test_smoke_entry_points():
    for contract_path in CONTRACT_PATHS:
        contract = load_contract(contract_path)
        exp_dir = contract_path.parent
        entry = exp_dir / contract["entry_point"]
        smoke = contract["smoke_test"]

        if smoke["kind"] == "help":
            proc = subprocess.run(
                [sys.executable, entry.name, *smoke.get("args", [])],
                cwd=exp_dir,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr or proc.stdout
            assert "usage" in proc.stdout.lower() or "usage" in proc.stderr.lower()
        elif smoke["kind"] == "import_main":
            spec = importlib.util.spec_from_file_location(f"smoke_{contract['id']}", entry)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            assert hasattr(module, "main")
            assert callable(module.main)
        else:
            raise AssertionError(f"Unknown smoke kind: {smoke['kind']}")
