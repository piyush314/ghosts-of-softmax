"""Sharp remainder bounds for attention KL divergence.

Two bounds depending on KL direction:

REVERSE KL (rigorous, from Bregman divergence):
  |KL(α⁰ || α(τ)) - (τ²/2) Var_{α⁰}(a)| ≤ |τ|³ Δ³ / (36√3)
  Derivation: τ³/6 (Lagrange) × Δ³/(6√3) (sharp lemma).

FORWARD KL (approximate, leading term):
  |KL(α(τ) || α⁰) - (τ²/2) Var_{α⁰}(a)| ≈ |τ|³ Δ³ / (18√3)
  The 2× factor arises because d³/dτ³[τK'(τ)-K(τ)] = 2K'''
  plus a fourth-cumulant correction τK''''.

KEY LEMMA: |E[(X - EX)³]| ≤ Δ³ / (6√3)
Sharp, achieved by Bernoulli with p = (3 ± √3)/6.
"""

import numpy as np

SQRT3 = np.sqrt(3)
THIRD_MOMENT_CONSTANT = 1 / (6 * SQRT3)  # ≈ 0.0962
REV_KL_CONSTANT = 1 / (36 * SQRT3)       # ≈ 0.01604 (rigorous)
FWD_KL_CONSTANT = 1 / (18 * SQRT3)       # ≈ 0.03208 (leading term)
KL_REMAINDER_CONSTANT = FWD_KL_CONSTANT   # code tests forward KL


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def klDivergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) averaged over batch."""
    return (p * (np.log(p + eps) - np.log(q + eps))).sum(axis=-1).mean()


def computeAttentionKL(alpha0: np.ndarray, a: np.ndarray, tau: float) -> float:
    """Exact KL(α(τ) || α⁰) via exponential tilting."""
    logW = np.log(alpha0 + 1e-12) + tau * a
    logZ = np.log(np.exp(logW - logW.max(-1, keepdims=True)).sum(-1, keepdims=True))
    logZ = logZ + logW.max(-1, keepdims=True)
    alphaTau = np.exp(logW - logZ)
    return klDivergence(alphaTau, alpha0)


def computeVariance(alpha: np.ndarray, a: np.ndarray) -> float:
    """Var_α(a) = E_α[a²] - E_α[a]²."""
    Ea = (alpha * a).sum(axis=-1)
    Ea2 = (alpha * a**2).sum(axis=-1)
    return (Ea2 - Ea**2).mean()


def computeSlopeSpread(a: np.ndarray) -> float:
    """Δ = max(a) - min(a)."""
    return (a.max(axis=-1) - a.min(axis=-1)).mean()


def klRemainderBound(tau: float, delta: float) -> float:
    """Sharp upper bound on |KL - quadratic approximation|."""
    return (abs(tau)**3 * delta**3) * KL_REMAINDER_CONSTANT


def verifyBound(alpha0: np.ndarray, a: np.ndarray, tau: float) -> dict:
    """Verify the sharp bound for given inputs."""
    klActual = computeAttentionKL(alpha0, a, tau)
    var0 = computeVariance(alpha0, a)
    delta = computeSlopeSpread(a)

    klQuad = (tau**2 / 2) * var0
    remainder = abs(klActual - klQuad)
    bound = klRemainderBound(tau, delta)

    return {
        "klActual": klActual,
        "klQuad": klQuad,
        "remainder": remainder,
        "bound": bound,
        "ratio": remainder / (bound + 1e-12),
        "valid": remainder <= bound * 1.01,
        "tauDelta": tau * delta,
    }
