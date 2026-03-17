# Ghosts of Softmax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig1.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Ghosts of Softmax: Complex Singularities That Limit Safe Step Sizes in Cross-Entropy**
> [[Paper (PDF)](assets/paper.pdf)] · [arXiv:2603.13552](https://arxiv.org/abs/2603.13552)

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

<img src="assets/infographic.png" alt="Ghosts of Softmax — overview" width="85%">

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

1. Learn the controller idea first.

   Start with [`tutorials/00_step_controller_intro.ipynb`](tutorials/00_step_controller_intro.ipynb)
   [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/tutorials/00_step_controller_intro.ipynb).
   This notebook shows, on a small digits classification problem, how a batch
   JVP estimate of `rho` changes a standard SGD step. It compares:
   - fixed SGD with three learning rates,
   - rho-capped SGD with the same three learning rates,
   - one rho-set SGD run driven entirely by local geometry.
   It first focuses on loss and accuracy, then shows how effective learning
   rate and normalized step size evolve during training.

2. Reproduce a headline result quickly.

   Start with [`notebooks/fig1.ipynb`](notebooks/fig1.ipynb), which reproduces
   the phase transition near `r = 1` in a lightweight setting.

3. Learn the ideas in order.

   Work through the [tutorials](#tutorials) in sequence.

4. Run the experiment scripts directly.

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
| Fig 10 variant | Exact actual-step Adam controller at 10000x spike | [`fig10_exact_actual_adam.ipynb`](notebooks/fig10_exact_actual_adam.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/fig10_exact_actual_adam.ipynb) | ~10 min |
| ResNet-18 | CIFAR-10 instability | [`resnet18.ipynb`](notebooks/resnet18.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/resnet18.ipynb) | ~30 min |
| Ghost envelope | Re/Im scatter + bands | [`ghostenvelope.ipynb`](notebooks/ghostenvelope.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piyush314/ghosts-of-softmax/blob/main/notebooks/ghostenvelope.ipynb) | ~10 min |

## Tutorials

| Notebook | Topic |
|----------|-------|
| [`00_step_controller_intro.ipynb`](tutorials/00_step_controller_intro.ipynb) | First tutorial: fixed SGD vs rho-capped SGD vs rho-set SGD |
| [`01_adam_controller.ipynb`](tutorials/01_adam_controller.ipynb) | Second tutorial: exact directional Adam controller vs earlier rho-scaled alternative |
| [`02_momentum_controller.ipynb`](tutorials/02_momentum_controller.ipynb) | Additional controller tutorial: exact momentum-SGD controller vs earlier rho-scaled alternative |
| [`03_binary_radius.ipynb`](tutorials/03_binary_radius.ipynb) | Binary softmax convergence radius |
| [`04_kl_bound.ipynb`](tutorials/04_kl_bound.ipynb) | KL divergence bound |

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

## Frequently Asked Questions

### The Core Idea

<details>
<summary><b>Q: The one-sentence version?</b></summary>

Cross-entropy loss has complex singularities that limit how far you
can trust its Taylor expansion. The safe step size is bounded by ρ_a
= π/Δ_a, where Δ_a measures the spread of logit directional
derivatives. This bound shrinks as training sharpens predictions,
even when the loss surface appears flat.
</details>

<details>
<summary><b>Q: Why should I care about complex singularities? My parameters are real.</b></summary>

Your parameters are real, but your optimizer's local model depends on
the loss's analytic structure in the complex plane. The Taylor series
of any analytic function converges only within a radius set by the
nearest complex singularity (Cauchy–Hadamard theorem). This is a
theorem, not an approximation. Beyond that radius, adding more Taylor
terms makes the approximation worse. Any descent argument that
extrapolates local derivatives assumes the step stays inside this
radius.
</details>

<details>
<summary><b>Q: How is this different from L-smoothness?</b></summary>

L-smoothness says the gradient doesn't change too fast, giving η <
2/L. The convergence radius says the Taylor series itself converges.
These constraints differ. For cross-entropy, the Hessian curvature
decays exponentially with margin (σ(1−σ) ~ e^{-δ}), so
curvature-based bounds become very permissive late in training.
Meanwhile, the convergence radius decays only algebraically (~ δ).
Late in training, the Hessian suggests huge steps while the complex
singularities forbid them. The mismatch can exceed 4000× at margin δ
= 10.
</details>

---

### The Linearization Question

<details>
<summary><b>Q: You linearize logits. Isn't that a severe approximation for deep networks?</b></summary>

Linearization is a presentation choice, not a theoretical bottleneck.
The core mechanism—complex zeros of the partition function cap the
convergence radius—holds for any parameterization. What changes is
how you locate those zeros.

Consider this hierarchy:

- **Linear logits** z_k(τ) = z_k(0) + a_k τ: The partition function
is an exponential polynomial Σ w_k exp(a_k τ). Its zeros give the
closed-form bound ρ_a = π/Δ_a. This is the paper's presentation.
- **Quadratic logits** z_k(τ) = z_k(0) + a_k τ + b_k τ²: The
partition function is Σ w_k exp(a_k τ + b_k τ²). Zeros exist but have
no universal closed form. The mechanism remains the same.
- **Padé logit model**: A rational approximation to the logit
trajectory leads to zeros of composed low-degree rational and
exponential functions. Again, the mechanism is unchanged.

In each case, you find zeros of an exponential sum or a mild
generalization. The linear case isolates the controlling variable Δ_a
in a single interpretable formula. The qualitative story—complex
zeros imply a finite radius, which constrains step size—survives
moving up the hierarchy.
</details>

<details>
<summary><b>Q: So what <i>would</i> actually break the theory?</b></summary>

The real threat is **non-analyticity of the logit map itself**. If
z_k(τ) is not analytic (e.g., it passes through a ReLU kink), then
there is no Taylor series to discuss. The framework then requires
reinterpretation.

In practice, ReLU networks have piecewise-linear logit trajectories.
Analyticity holds on each piece. The question becomes: which is
closer—the nearest ReLU kink or the nearest softmax ghost? Our
experiments suggest softmax ghosts are typically the binding
constraint in normalized networks, but this is empirical, not a
theorem.
</details>

---

### Activation Singularities

<details>
<summary><b>Q: Don't activations have their own singularities?</b></summary>

Yes. Appendix 2 catalogs them. Every non-entire activation function
introduces complex singularities that can cap the convergence radius
before softmax ghosts do. Ranking by singularity distance:

| Activation | Nearest singularity | Distance from real axis |
|---|---|---|
| Exact GELU (erf-based) | None (entire) | ∞ |
| SiLU/Swish, Softplus | iπ(2k+1) | π |
| Tanh | i(π/2 + πk) | π/2 |
| ReLU | Real axis (kinks) | 0 (at breakpoints) |

For gated FFNs, the same ranking applies to the gate: exact GeGLU >
SwiGLU > ReGLU.
</details>

<details>
<summary><b>Q: Wait, tanh-approximate GELU is worse than exact GELU?</b></summary>

Yes. Exact GELU uses the error function (erf), which is entire—it has
no finite complex singularities. The common tanh approximation
reintroduces tanh poles at imaginary distance π/2. A seemingly
harmless shortcut creates artificial singularities.

Many modern frameworks provide exact erf-based GELU, but legacy
codebases and some hardware paths still use the tanh approximation.
</details>

<details>
<summary><b>Q: Why not just use entire activations everywhere?</b></summary>

It's not that simple. "Entire" means no finite complex singularities,
but a nonconstant entire function must be unbounded on C. So
"infinite activation radius" doesn't guarantee arbitrarily large safe
steps—the softmax radius ρ_a and growth rate in the relevant complex
strip remain as constraints.

Practically, we cannot yet cleanly separate activation singularity
effects from other training factors (initialization, normalization,
optimizer dynamics, data distribution, etc.). The analysis is
intellectually coherent but hard to isolate experimentally. We
present the ranking as a structural observation, not a validated
recommendation.
</details>

<details>
<summary><b>Q: Tell me about RIA, GaussGLU, and analytic normalization.</b></summary>

These designs follow from the convergence-radius principle (Appendix
3).

**RIA (Rectified Integral Activation):** The integral of the Gaussian
CDF—ReLU convolved with a Gaussian. It's entire, strictly convex, has
a monotone derivative, and recovers ReLU as β → ∞. It resembles
softplus but without the iπ(2k+1) ghost lattice.

**GaussGLU:** Uses the Gaussian CDF Φ(βx) as the gate instead of
sigmoid. This is the radius-clean analogue of SwiGLU—SwiGLU inherits
logistic poles, GaussGLU has none.

**Analytic normalization:** LayerNorm and RMSNorm use 1/√v, which has
a branch point at v = 0. Applying the Weierstrass transform (Gaussian
convolution) yields an entire function expressible via parabolic
cylinder functions, removing the singularity.

These are theoretical suggestions. Whether they yield measurable
training stability gains is untested.
</details>

---

### Practical Usage

<details>
<summary><b>Q: How do I compute ρ_a for my model?</b></summary>

Use one Jacobian-vector product (JVP) per sample:

1. Pick your update direction v (normalized gradient, Adam direction,
etc.).
2. For each sample x in your batch, compute a(x; v) = J_z(x) · v.
3. Compute Δ_a(x; v) = max_k a_k - min_k a_k per sample.
4. ρ_a = π / max_x Δ_a(x; v).

The normalized step is r = τ / ρ_a, where τ = ‖p‖ is the step
distance. In experiments, r < 1 was always safe.
</details>

<details>
<summary><b>Q: What's the computational overhead?</b></summary>

On ResNet-18/CIFAR-10 (batch 128, RTX 6000 Ada):
- Baseline SGD step: 12.6 ms
- Finite-difference ρ_a estimation: 20.9 ms (+66%)
- Exact JVP: 28.7 ms (+129%)

Finite differences are approximate but cheaper. Overhead on large
models is unmeasured.
</details>

<details>
<summary><b>Q: How should I use the ρ_a-controller in practice?</b></summary>

Integrate it as a **replacement for gradient clipping** in your
existing setup. Keep your learning rate schedule and optimizer. After
computing the update p, check if ‖p‖ exceeds ρ_a and clip to ρ_a if
needed. This adapts to local geometry instead of using a fixed
threshold.

You can also use it alongside gradient clipping—they address
different failure modes. Gradient clipping bounds raw gradient
magnitude; ρ_a-clip bounds step distance relative to the convergence
radius.
</details>

<details>
<summary><b>Q: What target r should I use?</b></summary>

- **r = 0.5 is the recommended starting point.** It's definitively
safe and doesn't sacrifice convergence rate.
- **r = 1 is the theoretical boundary.** The paper's best ResNet-18
result used r = 1, but you're relying on slack.
- **r = 2 still works in most tested settings** due to slack, but
loses the guarantee.
- **r = 4 is where things start breaking** with high variance.

For reliable training, use r = 0.5. To push the boundary, use r = 1.
</details>

<details>
<summary><b>Q: The bound uses max over all samples. Isn't that outlier-dominated?</b></summary>

Yes. The dataset-wide bound ρ_a = π / max_x Δ_a(x; v) is set by the
single worst-case sample, which can be overly conservative.

A practical alternative is to use a **quantile** of Δ_a (e.g., 99th
or 95th percentile) instead of the maximum. The theory requires the
max, but in practice a single outlier may not destabilize the entire
batch update. The right aggregation is likely architecture- and
data-dependent.
</details>

<details>
<summary><b>Q: Can I estimate ρ_a from a subsample instead of the full dataset?</b></summary>

Yes. Computing JVP for every training sample every step is expensive.
Reasonable strategies:

1. **Current-batch estimation:** Compute Δ_a for every sample in the
current mini-batch and take the max (or a quantile). This is cheap
but may miss globally worst-case samples.
2. **Windowed min across an epoch:** Compute ρ_a per batch, then take
the running minimum over a sliding window spanning one full epoch.
This gives full dataset coverage without computing all samples every
step.
3. **Tracking the bottleneck set:** Maintain a small buffer of
samples that produced the smallest ρ_a in recent batches. Always
include them alongside the current batch when estimating the radius.
This assumes the bottleneck set is stable across training—an
empirical question.

The right strategy depends on how concentrated the Δ_a distribution
is.
</details>

<details>
<summary><b>Q: How conservative is the bound, really?</b></summary>

Two layers of conservatism:

1. **Within the linearized model:** ρ_a = π/Δ_a is the worst case
(balanced logits, δ = 0). The exact linearized radius is ρ* = √(δ² +
π²)/Δ_a, which is larger by √(1 + (δ/π)²). For δ = 10, it's ~3.3×
larger.
2. **Linearization vs reality:** The true logit trajectory z_k(τ) is
nonlinear. For networks with residual connections, the linearized
bound is close; for others, the gap could be larger.

Quantifying this precisely is an open research question.
</details>

<details>
<summary><b>Q: Should I use the ρ_a-controller as a standalone optimizer?</b></summary>

The paper's standalone controller is a proof of concept. It shows ρ_a
contains enough information to determine a safe step size without a
schedule. It hasn't been tested at scale.

The more practical near-term use is as a **diagnostic and safety
clip** within an existing setup.
</details>

<details>
<summary><b>Q: The controller reached 85.3% on ResNet-18/CIFAR-10 without a schedule. Is that good?</b></summary>

It's a proof of concept. The point is that a controller using only
local geometry outperformed the best fixed learning rate (82.6%) over
10 epochs. State-of-the-art results use longer training, data
augmentation, and tuned schedules—we stripped these to isolate the
mechanism.
</details>

---

### Scope and Limitations

<details>
<summary><b>Q: Does this explain all training instabilities?</b></summary>

No. It explains instabilities arising from step size interacting with
cross-entropy's analytic structure. Other sources (numerical
precision, BatchNorm corruption, data issues, etc.) are separate.
</details>

<details>
<summary><b>Q: Has this been tested at frontier scale?</b></summary>

Not yet. Experiments use small models (up to ResNet-18 on CIFAR-10
and tiny transformers). The mechanism is architecture-independent, so
the question is whether it's the binding constraint at scale or if
something else breaks first. This is open.

For large-vocabulary models (50,000+ classes), Δ_a is generally
larger, making ρ_a smaller. Whether LLMs operate at r ≫ 1 through
favorable cancellation is unknown.
</details>

<details>
<summary><b>Q: What about multi-step dynamics?</b></summary>

The bounds describe one-step reliability. A run can survive
occasional r > 1 updates if the iteration self-corrects. Divergence
required r > 1 consistently over several iterations.

The controller recomputes ρ_a at every step, adapting as ρ_a changes.
This is iterative one-step safety, not global convergence.
</details>

<details>
<summary><b>Q: The bound is conservative. Doesn't that limit its usefulness?</b></summary>

The bound ρ_a = π/Δ_a is the worst case. The exact binary radius ρ* =
√(δ² + π²)/Δ_a exceeds ρ_a by √(1 + (δ/π)²). Confident predictions
have more headroom.

This conservatism is by design—a lower bound should not over-promise
safety. In practice, most architectures survive r > 1 due to this
slack. The bound tells you where safety is guaranteed, not where
failure is certain.

Tightening the bound sample-by-sample using the logit gap δ is a
natural refinement.
</details>

---

### Connections

<details>
<summary><b>Q: How does this relate to the edge of stability?</b></summary>

Cohen et al. (2021) observe that gradient descent keeps λ_max(H) ≈
2/η. Our framework offers a complementary view: as training
progresses, margins grow, Hessian curvature drops, but the
convergence radius also shrinks. The edge of stability describes what
the Hessian does at equilibrium; the convergence radius is a separate
constraint the Hessian doesn't see.

Appendix 5 shows the crossover margin where the ghost constraint
becomes stricter satisfies δ* ≈ A + B ln η, consistent with an
edge-of-stability mechanism.
</details>

<details>
<summary><b>Q: How does this relate to loss spikes in large-scale training (PaLM, LLaMA)?</b></summary>

Our interpretation: these reflect steps exceeding a shrinking ρ_a. As
training sharpens predictions, Δ_a grows, the safe step size
contracts, and a previously fine learning rate may violate r < 1.
This matches spikes appearing later in training.

This is an interpretation, not a validated explanation. Testing it
requires computing r during a frontier training run.
</details>

<details>
<summary><b>Q: What's the connection to Lee-Yang zeros and statistical mechanics?</b></summary>

The softmax partition function F(τ) = Σ w_k exp(a_k τ) is identical
to a statistical mechanics partition function with states k, energies
-a_k, and degeneracies w_k at inverse temperature τ. Its complex
zeros are analogous to Lee-Yang zeros—points where phase transitions
occur.

Lee and Yang (1952) showed Ising model zeros lie on the unit circle.
Our setting differs, but the insight transfers: zeros of the
partition function, even off the real axis, constrain real-axis
behavior. Here, they constrain the Taylor expansion's convergence
radius.
</details>

---

### Paper-Specific

<details>
<summary><b>Q: Why "ghosts"?</b></summary>

The complex zeros of the partition function are invisible on the real
line but haunt the Taylor series, limiting its convergence radius.
"Ghosts" is more evocative than "complex partition-function zeros."
</details>

<details>
<summary><b>Q: Is the π in ρ_a = π/Δ_a deep or coincidental?</b></summary>

It's exact. For two exponentials to cancel, e^{iπ} = -1 (Euler's
identity). For Σ w_k exp(a_k τ) to have a zero, terms must cancel,
requiring a phase difference of π between fastest- and
slowest-growing exponentials. At imaginary step size y, the phase
spread is Δ_a · |y|. Setting Δ_a · |y| = π gives |y| = π/Δ_a. The π
is Euler's π.
</details>

---

### What This Paper Is and Isn't

This paper is an **initial point result**: it identifies the
convergence-radius constraint, shows its origin, derives a computable
bound, and demonstrates predictive power in controlled settings.

It is **not** about:

- **The optimal way to compute ρ_a.** Subsampling strategies,
windowed estimates, quantile vs max—these are practical engineering
questions that remain open.
- **The best logit model for the bound.** Linear logits give π/Δ_a.
Higher-order models give tighter bounds at higher cost. The paper
presents the simplest member for clarity.
- **A production optimizer.** The ρ_a-controller is a proof of
concept. Integration with Adam, modern schedulers, and large-scale
workloads is future engineering.
- **A complete theory of training stability.** The bound describes
one-step reliability for softmax/cross-entropy. Activation
singularities, normalization, multi-step dynamics, gradient noise,
and data effects are separate.

Many open questions follow naturally: How tight is the bound for a
given architecture? What's the right subsampling strategy? Does the
bottleneck set change during training? Answering these requires
systematic experimentation beyond a single paper.

The core contribution is identifying the mechanism and demonstrating
it's real, computable, and predictive. Everything else is downstream.

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
