# Ghosts of Softmax

Reproducibility code for the paper:
**Ghosts of Softmax: When Complex Zeros Cap the Convergence Radius**

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
pytest
```

## Tested baseline

The current checked baseline is:

- Python `3.12`
- PyTorch `2.10.0`
- torchvision `0.25.0`
- NumPy `1.26.4`
- Matplotlib `3.10.8`
- scikit-learn `1.5.2`

For a pinned install set, see `requirements.txt`.

## Validation

Unit and contract tests:

```bash
pytest
```

End-to-end smoke runs for selected experiments:

```bash
GHOSTS_RUN_SMOKE=1 pytest tests/test_smoke_runs.py -q
```

Notebook execution:

```bash
jupyter nbconvert --to notebook --execute tutorials/01_binary_radius.ipynb --output /tmp/01_binary_radius.executed.ipynb
jupyter nbconvert --to notebook --execute tutorials/02_kl_bound.ipynb --output /tmp/02_kl_bound.executed.ipynb
jupyter nbconvert --to notebook --execute tutorials/03_rho_controller.ipynb --output /tmp/03_rho_controller.executed.ipynb
```

## Experiments

Each experiment has a canonical `run.py`. Some also provide separate plotting
entry points:

| Directory | Paper result |
|-----------|-------------|
| `experiments/phasetransition/` | Phase transition at r = 1 |
| `experiments/randomdirs/` | Random-direction validation |
| `experiments/lrspike/` | LR spike controller comparison |
| `experiments/tempfingerprint/` | Temperature-scaling fingerprint |
| `experiments/archgrid/` | Cross-architecture spike comparison |
| `experiments/tfmbottlenecks/` | Transformer bottleneck analysis |
| `experiments/resnetnatural/` | ResNet-18/CIFAR-10 natural instability |

Example:

```bash
python experiments/phasetransition/run.py
python experiments/phasetransition/plot.py
```

For experiments without a dedicated `plot.py`, the experiment README is the
source of truth for expected outputs and how results are summarized.

## Core library

`src/ghosts/` contains reusable modules:

- `radii.py` — softmax convergence radius computation
- `control.py` — rho-adaptive controller
- `models.py` — small transformer
- `theory.py` — KL divergence bound
- `hooks.py` — model instrumentation

## License

MIT
