"""Core margin (rho) measurement functions for stability analysis."""

import torch
import torch.nn as nn
from typing import Optional
import math

PI = math.pi


def compute_logit_gap(
    logits: torch.Tensor,
    gap: str = "maxmin",
    reduce: str = "mean",
):
    """Compute logit gap (max-min or top-two) with configurable reduction.

    Args:
        logits: (batch, vocab) or (batch, seq, vocab)
        gap: "maxmin" or "top2"
        reduce: "mean", "max", "per_sample", or "none"

    Returns:
        Scalar float for mean/max, tensor for per_sample/none.
    """
    if gap == "top2":
        top2 = torch.topk(logits, k=2, dim=-1).values
        gapVals = top2[..., 0] - top2[..., 1]
    else:
        gapVals = logits.max(dim=-1).values - logits.min(dim=-1).values

    if reduce == "none":
        return gapVals
    if reduce == "per_sample":
        if gapVals.dim() == 1:
            return gapVals
        return gapVals.mean(dim=-1)
    if reduce == "max":
        return gapVals.max().item()
    return gapVals.mean().item()


def compute_rho_from_delta(
    delta,
    eps: float = 1e-6,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    inf_if_small: bool = False,
):
    """Compute rho = pi/delta with consistent edge handling."""
    if torch.is_tensor(delta):
        if inf_if_small:
            rho = torch.where(delta <= eps, torch.tensor(float("inf"), device=delta.device), PI / (delta + eps))
        else:
            rho = PI / (delta + eps)
        if cap is not None:
            rho = torch.clamp(rho, max=cap)
        if floor is not None:
            rho = torch.clamp(rho, min=floor)
        return rho

    if inf_if_small and delta <= eps:
        return float("inf")
    rho = PI / (delta + eps)
    if cap is not None:
        rho = min(rho, cap)
    if floor is not None:
        rho = max(rho, floor)
    return rho


def compute_rho_out(
    logits: torch.Tensor,
    gap: str = "maxmin",
    reduce: str = "mean",
    eps: float = 1e-6,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    inf_if_small: bool = False,
):
    """Compute rho_out from logits with configurable gap/reduction."""
    delta = compute_logit_gap(logits, gap=gap, reduce=reduce)
    return compute_rho_from_delta(delta, eps=eps, cap=cap, floor=floor, inf_if_small=inf_if_small)


def measureRhoFFN(
    model: nn.Module,
    inputIds: torch.Tensor,
    lossFunc: Optional[callable] = None,
) -> float:
    """
    Measure FFN stability margin via gradient norm proxy.

    rho_FFN ≈ pi/2 / ||grad||

    Lower rho = less stable (closer to blowup).
    """
    model.zero_grad()

    if lossFunc is None:
        outputs = model(inputIds, labels=inputIds)
        loss = outputs.loss
    else:
        loss = lossFunc(model, inputIds)

    loss.backward()

    gradNorm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gradNorm += p.grad.norm().item() ** 2
    gradNorm = gradNorm ** 0.5

    model.zero_grad()

    if gradNorm < 1e-10:
        return float("inf")

    return (PI / 2) / gradNorm


def measureRhoAttn(
    attentions: list[torch.Tensor],
    mode: str = "entropy",
) -> float:
    """
    Measure attention stability margin from attention weights.

    Args:
        attentions: List of attention tensors (batch, heads, seq, seq)
        mode: "entropy" or "spread"

    Returns:
        Minimum rho across all attention heads
    """
    minRho = float("inf")

    for layerAttn in attentions:
        bsz, nHeads, seqLen, _ = layerAttn.shape

        for h in range(nHeads):
            attn = layerAttn[:, h, :, :]  # (batch, seq, seq)

            if mode == "entropy":
                entropy = -(attn * (attn + 1e-10).log()).sum(-1)
                minEntropy = entropy.min().item()
                delta = 2.0 / (minEntropy + 0.1)
            else:
                maxVal = attn.max(dim=-1).values
                minVal = attn.min(dim=-1).values
                delta = (maxVal - minVal).max().item()
                delta = max(delta, 1e-10)

            rho = PI / delta
            minRho = min(minRho, rho)

    return minRho


def measureRhoOut(logits: torch.Tensor) -> float:
    """
    Measure output stability margin from logits.

    rho_out = pi / Delta_out where Delta_out = max - min logit
    """
    return compute_rho_out(logits, gap="maxmin", reduce="max", eps=1e-10, inf_if_small=True)


def measureRhoNet(
    model: nn.Module,
    inputIds: torch.Tensor,
    lossFunc: Optional[callable] = None,
) -> dict:
    """
    Compute all three stability margins and return the minimum.

    Returns dict with rhoFFN, rhoAttn, rhoOut, rhoNet (min of all).
    """
    model.eval()

    with torch.no_grad():
        outputs = model(inputIds, output_attentions=True)
        rhoAttn = measureRhoAttn(outputs.attentions)
        rhoOut = measureRhoOut(outputs.logits)

    model.train()
    rhoFFN = measureRhoFFN(model, inputIds, lossFunc)
    model.eval()

    return {
        "rhoFFN": rhoFFN,
        "rhoAttn": rhoAttn,
        "rhoOut": rhoOut,
        "rhoNet": min(rhoFFN, rhoAttn, rhoOut),
    }
