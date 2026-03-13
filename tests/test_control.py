"""Tests for ghosts.control — rho-adaptive optimizer."""

import math
import torch
import torch.nn as nn
import pytest
from ghosts.control import computeRho, RhoScaledAdam

PI = math.pi


def test_compute_rho_basic():
    logits = torch.tensor([[0.0, 2.0, 1.0]])
    rho = computeRho(logits, method="minmax")
    expected = PI / (2.0 + 1e-6)
    assert abs(rho - expected) < 1e-4


def test_rho_scaled_adam_step():
    model = nn.Linear(4, 3)
    opt = RhoScaledAdam(model.parameters(), lr=0.01)
    x = torch.randn(2, 4)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    opt.step(logits=logits)
    stats = opt.getStats()
    assert stats["rho"] > 0
    assert stats["effectiveLR"] > 0
    assert stats["effectiveLR"] <= stats["baseLR"] * 1.0 + 1e-8


def test_rho_scaled_adam_caps():
    model = nn.Linear(4, 3)
    opt = RhoScaledAdam(
        model.parameters(), lr=0.01, rhoCap=0.5, rhoFloor=0.01,
    )
    x = torch.randn(2, 4)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    opt.step(logits=logits)
    stats = opt.getStats()
    assert stats["effectiveLR"] <= 0.01 * 0.5 + 1e-8


def test_rho_scaled_adam_explicit_rho():
    model = nn.Linear(4, 3)
    opt = RhoScaledAdam(model.parameters(), lr=0.01, rhoCap=10.0)
    x = torch.randn(2, 4)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    opt.step(rho=0.3)
    stats = opt.getStats()
    assert abs(stats["rho"] - 0.3) < 1e-8
    assert abs(stats["effectiveLR"] - 0.01 * 0.3) < 1e-8
