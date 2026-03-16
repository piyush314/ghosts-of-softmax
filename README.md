# Ghosts of Softmax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig1.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Ghosts of Softmax: When Complex Zeros Cap the Convergence Radius**
> [[arXiv]](https://arxiv.org/abs/2503.XXXXX)

The softmax partition function has complex zeros that cap the Taylor
convergence radius of cross-entropy loss. Beyond that radius, local
surrogates need not track the true loss, so descent guarantees become
unreliable. We derive closed-form expressions for this radius and show
that the normalized step size r = τ/ρ\_a cleanly separates safe from
dangerous updates across six architectures. A controller enforcing
τ ≤ ρ\_a survives 10,000× LR spikes where gradient clipping collapses.

![Phase transition at r = 1](assets/teaser.png)
*Test accuracy retained after one gradient step. All architectures
collapse once the normalized step r = τ/ρ\_a exceeds 1.*

## Install

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

## Start Here

Use the repo in one of three ways:

1. Reproduce a headline result quickly.

   Start with [`notebooks/fig1.ipynb`](notebooks/fig1.ipynb), which reproduces
   the phase transition at `r = \tau / \rho_a \approx 1` in a lightweight
   setting.

2. Learn the ideas in order.

   Work through the tutorials:

   - [`tutorials/01_binary_radius.ipynb`](tutorials/01_binary_radius.ipynb)
   - [`tutorials/02_kl_bound.ipynb`](tutorials/02_kl_bound.ipynb)
   - [`tutorials/03_rho_controller.ipynb`](tutorials/03_rho_controller.ipynb)

3. Run the experiment scripts directly.

   Start with [`experiments/phasetransition/run.py`](experiments/phasetransition/run.py)
   or [`experiments/lrspike/run.py`](experiments/lrspike/run.py), then use the
   experiment contracts and READMEs to scale up to the full paper runs.

## Validation

Check the install before running heavier experiments:

```bash
pytest
GHOSTS_RUN_SMOKE=1 pytest tests/test_smoke_runs.py -q
```

## Reproduce Paper Figures

| Figure | Description | Notebook | Colab | Runtime |
|--------|-------------|----------|-------|---------|
| Fig 1 | Phase transition | [`fig1.ipynb`](notebooks/fig1.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig1.ipynb) | ~5 min |
| Fig 7 | JVP phase transition | [`fig7.ipynb`](notebooks/fig7.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig7.ipynb) | ~5 min |
| Fig 9 | Temperature fingerprint | [`fig9.ipynb`](notebooks/fig9.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig9.ipynb) | ~10 min |
| Fig 10 | Architecture grid | [`fig10.ipynb`](notebooks/fig10.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig10.ipynb) | ~15 min |
| ResNet-18 | CIFAR-10 instability | [`resnet18.ipynb`](notebooks/resnet18.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/resnet18.ipynb) | ~30 min |

## Tutorials

| Notebook | Topic |
|----------|-------|
| [`01_binary_radius.ipynb`](tutorials/01_binary_radius.ipynb) | Binary softmax convergence radius |
| [`02_kl_bound.ipynb`](tutorials/02_kl_bound.ipynb) | KL divergence bound |
| [`03_rho_controller.ipynb`](tutorials/03_rho_controller.ipynb) | Rho-adaptive controller |

## Experiments

Each experiment has a canonical `run.py`. Some also include separate plotting
scripts or notebook entry points; the experiment README and `contract.json` are
the source of truth for outputs and reproduction commands.

| Directory | Paper result |
|-----------|-------------|
| `experiments/phasetransition/` | Phase transition at r = 1 |
| `experiments/lrspike/` | LR spike controller comparison |
| `experiments/tempfingerprint/` | Temperature-scaling fingerprint |
| `experiments/archgrid/` | Cross-architecture spike comparison |
| `experiments/tfmbottlenecks/` | Transformer bottleneck analysis |
| `experiments/resnetnatural/` | ResNet-18/CIFAR-10 natural instability |
| `experiments/randomdirs/` | Random-direction validation |

```bash
python experiments/phasetransition/run.py
```

## Core Library

`src/ghosts/` contains reusable modules:

- `radii.py` — softmax convergence radius computation
- `control.py` — rho-adaptive controller
- `models.py` — small transformer
- `theory.py` — KL divergence bound
- `hooks.py` — model instrumentation

## Citation

```bibtex
@article{ghosts2025,
  title   = {Ghosts of Softmax: When Complex Zeros Cap the Convergence Radius},
  author  = {Piyush Kumar},
  year    = {2026},
  journal = {arXiv preprint arXiv:2503.XXXXX},
}
```

## License

MIT
