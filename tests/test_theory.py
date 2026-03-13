"""Tests for ghosts.theory — KL divergence bounds."""

import numpy as np
import pytest
from ghosts.theory import (
    softmax,
    klDivergence,
    computeAttentionKL,
    computeVariance,
    computeSlopeSpread,
    klRemainderBound,
    verifyBound,
    KL_REMAINDER_CONSTANT,
)


def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0]])
    p = softmax(x)
    assert abs(p.sum() - 1.0) < 1e-10


def test_softmax_uniform():
    x = np.array([[0.0, 0.0, 0.0]])
    p = softmax(x)
    assert np.allclose(p, 1.0 / 3)


def test_kl_zero_same():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    kl = klDivergence(p, p)
    assert abs(kl) < 1e-10


def test_kl_positive():
    p = softmax(np.array([[1.0, 0.0, 0.0]]))
    q = softmax(np.array([[0.0, 1.0, 0.0]]))
    kl = klDivergence(p, q)
    assert kl > 0


def test_variance_zero_uniform():
    alpha = np.array([[0.25, 0.25, 0.25, 0.25]])
    a = np.array([[1.0, 1.0, 1.0, 1.0]])
    var = computeVariance(alpha, a)
    assert abs(var) < 1e-10


def test_slope_spread():
    a = np.array([[1.0, 5.0, 3.0]])
    delta = computeSlopeSpread(a)
    assert abs(delta - 4.0) < 1e-10


def test_remainder_bound_scales_cubically():
    b1 = klRemainderBound(1.0, 1.0)
    b2 = klRemainderBound(2.0, 1.0)
    assert abs(b2 / b1 - 8.0) < 1e-10


def test_verify_bound_holds():
    """The sharp bound should hold for random inputs."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(3, 10)
        z = rng.standard_normal((1, n))
        a = rng.standard_normal((1, n))
        alpha0 = softmax(z)
        tau = rng.uniform(0.01, 0.5)
        result = verifyBound(alpha0, a, tau)
        assert result["valid"], (
            f"Bound violated: remainder={result['remainder']:.6e}, "
            f"bound={result['bound']:.6e}"
        )


def test_verify_bound_tight():
    """For Bernoulli extremum, the bound should be nearly tight."""
    p = (3 + np.sqrt(3)) / 6
    alpha0 = np.array([[p, 1 - p]])
    a = np.array([[0.0, 1.0]])
    tau = 0.1
    result = verifyBound(alpha0, a, tau)
    assert result["ratio"] > 0.5, "Bound should be reasonably tight"
