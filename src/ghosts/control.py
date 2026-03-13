"""ρ-scaled optimizer: adaptive learning rate based on analytic radius.

The key insight: instability occurs when η‖∇f‖/ρ > 1, where ρ = π/Δ
is the stability margin and Δ is the output logit spread.

Usage:
    from ghosts.control import RhoScaledAdam

    opt = RhoScaledAdam(model.parameters(), lr=3e-4)

    for batch in dataloader:
        logits = model(batch)
        loss = criterion(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step(logits)  # Pass logits to compute ρ
"""

import torch
from torch.optim import Optimizer
from ghosts.radii import compute_rho_out
from typing import Optional


def computeRho(logits: torch.Tensor, method: str = "minmax") -> float:
    """Compute stability margin ρ = π/Δ from output logits."""
    gap = "top2" if method == "top2" else "maxmin"
    return compute_rho_out(logits, gap=gap, reduce="mean", eps=1e-6)


def resolveRho(logits, rho, method):
    """Resolve ρ from logits or pre-computed value."""
    if rho is not None:
        return rho
    if logits is not None:
        with torch.no_grad():
            return computeRho(logits, method=method)
    return 1.0


class RhoScaledAdam(Optimizer):
    """Adam optimizer with ρ-scaled learning rate for stability.

    The effective learning rate is: lr_eff = lr * min(ρ, rhoCap)

    This prevents the instability ratio η‖∇f‖/ρ from exceeding 1
    by reducing LR when the model approaches instability (low ρ).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, rhoCap=1.0, rhoFloor=0.01,
                 rhoMethod="minmax"):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rhoCap=rhoCap, rhoFloor=rhoFloor)
        super().__init__(params, defaults)
        self.rhoMethod = rhoMethod
        self.lastRho = 1.0
        self.lastEffLR = lr

    def step(self, logits=None, rho=None):
        """Perform optimization step with ρ-scaled LR."""
        currentRho = resolveRho(logits, rho, self.rhoMethod)
        self.lastRho = currentRho

        for group in self.param_groups:
            scale = max(min(currentRho, group['rhoCap']), group['rhoFloor'])
            effLR = group['lr'] * scale
            self.lastEffLR = effLR

            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']

                stepSize = effLR / bc1
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                p.data.addcdiv_(m, denom, value=-stepSize)

    def getStats(self):
        """Return current optimizer statistics."""
        return {
            "rho": self.lastRho,
            "effectiveLR": self.lastEffLR,
            "baseLR": self.param_groups[0]['lr'],
            "scale": self.lastEffLR / self.param_groups[0]['lr'],
        }
