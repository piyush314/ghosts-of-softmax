# Ghosts of Softmax

Reproducibility code for the paper:
**Ghosts of Softmax: When Complex Zeros Cap the Convergence Radius**

## Quick start

```bash
pip install -e .
```

## Experiments

Each experiment has a `run.py` and `plot.py`:

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

## Core library

`src/ghosts/` contains reusable modules:

- `radii.py` — softmax convergence radius computation
- `control.py` — rho-adaptive controller
- `models.py` — small transformer
- `theory.py` — KL divergence bound
- `hooks.py` — model instrumentation

## License

MIT
