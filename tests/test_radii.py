"""Tests for ghosts.radii — softmax convergence radius computation."""

import math
import torch
import pytest
from ghosts.radii import (
    compute_logit_gap,
    compute_rho_from_delta,
    compute_rho_out,
)

PI = math.pi


def test_logit_gap_maxmin():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    gap = compute_logit_gap(logits, gap="maxmin", reduce="mean")
    assert abs(gap - 2.0) < 1e-6


def test_logit_gap_top2():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    gap = compute_logit_gap(logits, gap="top2", reduce="mean")
    assert abs(gap - 2.0) < 1e-6


def test_logit_gap_per_sample():
    logits = torch.tensor([[0.0, 4.0], [1.0, 2.0]])
    gaps = compute_logit_gap(logits, gap="maxmin", reduce="per_sample")
    assert gaps.shape == (2,)
    assert abs(gaps[0].item() - 4.0) < 1e-6
    assert abs(gaps[1].item() - 1.0) < 1e-6


def test_rho_from_delta_basic():
    delta = 1.0
    rho = compute_rho_from_delta(delta, eps=0.0)
    assert abs(rho - PI) < 1e-6


def test_rho_from_delta_tensor():
    delta = torch.tensor([1.0, 2.0])
    rho = compute_rho_from_delta(delta, eps=0.0)
    assert abs(rho[0].item() - PI) < 1e-5
    assert abs(rho[1].item() - PI / 2) < 1e-5


def test_rho_from_delta_cap():
    rho = compute_rho_from_delta(0.01, eps=0.0, cap=10.0)
    assert rho <= 10.0


def test_rho_from_delta_floor():
    rho = compute_rho_from_delta(1000.0, eps=0.0, floor=0.1)
    assert rho >= 0.1


def test_rho_from_delta_inf():
    rho = compute_rho_from_delta(0.0, eps=0.0, inf_if_small=True)
    assert rho == float("inf")


def test_rho_out_matches_manual():
    logits = torch.tensor([[0.0, PI]])  # delta = pi, rho = 1
    rho = compute_rho_out(logits, gap="maxmin", reduce="mean", eps=0.0)
    assert abs(rho - 1.0) < 1e-5


def test_rho_decreases_with_spread():
    narrow = torch.tensor([[1.0, 2.0]])
    wide = torch.tensor([[0.0, 10.0]])
    r1 = compute_rho_out(narrow, gap="maxmin", reduce="mean", eps=0.0)
    r2 = compute_rho_out(wide, gap="maxmin", reduce="mean", eps=0.0)
    assert r1 > r2


def test_binary_exact_radius():
    """For binary, rho* = sqrt(delta^2 + pi^2) / Delta_a."""
    delta = 2.0  # logit gap
    w1 = math.exp(delta / 2)
    w2 = math.exp(-delta / 2)
    da = 1.0  # Delta_a
    expected = math.sqrt(delta**2 + PI**2) / da
    # compute_rho_out gives pi/Delta_a (the lower bound)
    rho = compute_rho_out(
        torch.tensor([[delta / 2, -delta / 2]]),
        gap="maxmin", reduce="mean", eps=0.0,
    )
    assert rho <= expected + 1e-5  # lower bound <= exact
