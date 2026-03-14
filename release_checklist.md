# Release Checklist

This checklist is for the public `ghosts-of-softmax` repo, not the paper repo.
It is organized by release priority.

## Blocking

- [ ] Clean-clone install works:
  `python -m pip install -e .`
- [ ] Test collection works from a clean clone:
  `pytest`
- [ ] Fix package import path so tests can import `ghosts.*` without local hacks.
- [ ] README matches reality:
  do not claim every experiment has both `run.py` and `plot.py` unless that is true.
- [ ] Add `requirements.txt` as a pinned environment export from the tested setup.
- [ ] Pin one tested baseline stack in docs:
  Python version, PyTorch version, torchvision version.
- [ ] Replace broad dependency ranges in `pyproject.toml` with the tested release target or document the exact tested pair separately.
- [ ] Remove or fix any experiment entry point that is known to be stale, incomplete, or paper-inconsistent.
- [ ] Verify every released tutorial notebook runs top-to-bottom in a fresh environment.
- [ ] Add a CI workflow that at minimum runs install + tests on the supported Python version.

## Before v1 Tag

- [ ] Add one smoke test per released experiment:
  import, argument parsing, tiny run, expected output files.
- [ ] Add one reproducibility contract per experiment:
  command, config, seeds, dataset, outputs, expected metrics or tolerance bands.
- [ ] Add `plot.py` entry points where the README promises them, or reduce the promise.
- [ ] Standardize output layout for experiments:
  `results/<experiment>/<run_id>/` with config copy and summary file.
- [ ] Write one machine-readable summary per run:
  `summary.json` or `metrics.csv`.
- [ ] Include run metadata in summaries:
  git SHA, seed, config name, package versions, device, runtime.
- [ ] Add dataset provenance notes:
  source, preprocessing, and hash/count when feasible.
- [ ] Add a short README section on supported hardware:
  CPU/GPU expectations and approximate runtime class.
- [ ] Validate that the three released tutorials are thin wrappers over library/experiment code, not separate implementations.
- [ ] Add `CITATION.cff`.
- [ ] Add a top-level “reproduce the paper” section to `README.md`.
- [ ] Add one README per experiment with:
  purpose, command, outputs, expected runtime, expected hardware.

## Post v1

- [ ] Add notebook execution to CI, or convert notebooks to tested scripts plus rendered docs.
- [ ] Add a small `data/manifests/` directory with dataset manifests and checksums.
- [ ] Add release artifacts or badges:
  version tag, DOI if applicable, CI status.
- [ ] Revisit optional experiments deferred from v1:
  WikiText-2 / nanoGPT path, theory sanity experiments, Hessian-mismatch plots.
- [ ] Decide whether to ship small demo outputs for quick verification.
- [ ] Add API docs for `ghosts.radii`, `ghosts.control`, and `ghosts.theory`.
- [ ] Add contribution guidelines if the repo is meant to accept outside PRs.

## Acceptance Gate

Before calling the repo publicly releaseable, all of the following should be true:

- [ ] Fresh clone installs with one documented command.
- [ ] `pytest` passes.
- [ ] README instructions work as written.
- [ ] Each v1 experiment has one canonical run command and one canonical output summary.
- [ ] Each released notebook executes cleanly in the documented environment.
- [ ] No known paper-facing experiment is relying on hidden local state or undocumented cached artifacts.
