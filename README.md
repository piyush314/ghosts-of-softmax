# Ghosts of Softmax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig1.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Ghosts of Softmax: Complex Singularities That Limit Safe Step Sizes in Cross-Entropy**
> [[arXiv]](https://arxiv.org/abs/2503.XXXXX)

Large-scale training still suffers abrupt instabilities despite
decades of work on loss geometry, curvature, and optimizer design. A
common approach is to analyze a local Taylor surrogate of the loss
and derive conditions for its decrease. But a basic gap remains: a
Taylor expansion is only meaningful inside its radius of convergence.
Inside that radius, local surrogate arguments can be informative.
Outside it, the Taylor series no longer tracks the true function, so
guarantees for the surrogate may not apply to the actual loss.

This issue is easy to miss in deep learning because the radius of
convergence is hard to access from the real line. The standard way
to control Taylor convergence is to bound the full infinite series,
for instance through derivative growth rates or comparison to a
geometric series. For neural networks, even first- and second-order
information is expensive and incomplete, making direct reasoning
about higher derivatives impractical. As a result, optimization
analyses often rely on low-order local models without checking if
the underlying Taylor series is valid at the proposed step.

This work takes a different route. Instead of estimating convergence
from higher real derivatives, we use the Cauchy--Hadamard viewpoint:
the convergence radius of a Taylor series is set by the nearest
singularity in the complex plane. For cross-entropy training, these
singularities arise from complex zeros of the softmax partition
function. Under logit linearization, this leads to closed-form and
interpretable radius estimates. Specifically, the paper shows how to
bound the convergence radius from directional logit derivatives, and
why this provides a safety criterion fundamentally different from
Hessian-based smoothness.

To make this analysis concrete and reproducible, this repository
provides tutorials, notebooks, and experiment scripts for:
- Estimating the convergence radius using finite differences or
  Jacobian--vector products.
- Reproducing the main figures from the paper.
- Incorporating a radius-based step-size controller into practical
  optimizers (SGD, momentum SGD, Adam).

The controller's idea is simple: a proposed update step should not
exceed the local convergence radius. If the underlying optimizer
proposes a larger step, the controller rescales it to remain inside
the safe region suggested by the theory.

<img src="assets/teaser.png" alt="Phase transition at r = 1" width="75%">
*Test accuracy retained after one gradient step. All architectures
collapse once the normalized step r = τ/ρ_a exceeds 1.*

## Install

Requires Python 3.10+. GPU optional but recommended for experiment scripts.

```bash
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

## Start Here

Use the repo in one of three ways:

1. Reproduce a headline result quickly.

   Start with [`notebooks/fig1.ipynb`](notebooks/fig1.ipynb), which reproduces
   the phase transition near r = 1 in a lightweight setting.

2. Learn the ideas in order.

   Work through the [tutorials](#tutorials) in sequence.

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
| Ghost envelope | Re/Im scatter + bands | [`ghostenvelope.ipynb`](notebooks/ghostenvelope.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/ghostenvelope.ipynb) | ~10 min |

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
| `experiments/ghostenvelope/` | Ghost envelope visualization |
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
@article{ghosts2026,
  title   = {Ghosts of Softmax: Complex Singularities That Limit Safe Step Sizes in Cross-Entropy},
  author  = {Piyush Sao},
  year    = {2026},
  journal = {arXiv preprint arXiv:2503.XXXXX},
}
```

## License

MIT
