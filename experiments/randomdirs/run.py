#!/usr/bin/env python3
"""Random-direction sweep with corrected math and a different network.

Changes vs the original numpy prototype:
1) Uses exact autograd gradients/HVP (no manual backprop bug around LayerNorm).
2) Uses exact JVP for rho_a along arbitrary parameter directions.
3) Uses dynamic parameter shapes (no hardcoded flat shape assumptions).
4) Treats non-positive directional curvature as "no finite Hessian boundary".

Network used here is intentionally different from the original MLP: a linear classifier.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

FIG_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_digits_csv() -> Tuple[np.ndarray, np.ndarray]:
    import site
    candidates = []
    for sp in site.getsitepackages():
        candidates.append(Path(sp) / "sklearn" / "datasets" / "data" / "digits.csv.gz")
    usp = site.getusersitepackages()
    if usp:
        candidates.append(Path(usp) / "sklearn" / "datasets" / "data" / "digits.csv.gz")
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Could not find sklearn digits.csv.gz")
    arr = np.loadtxt(gzip.open(csv_path, "rt"), delimiter=",", dtype=np.float32)
    X = arr[:, :-1] / 16.0
    y = arr[:, -1].astype(np.int64)
    return X, y


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.30, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(test_ratio * len(idx)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class LinearNet(nn.Module):
    def __init__(self, D: int, C: int):
        super().__init__()
        self.fc = nn.Linear(D, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def named_params(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def copy_param_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def model_metrics(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        z = model(X)
        loss = float(F.cross_entropy(z, y).item())
        acc = float((z.argmax(dim=1) == y).float().mean().item())
    return acc, loss


def train_stage_linear(
    D: int,
    C: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    n_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 17)

    model = LinearNet(D, C)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(X_train)

    for _ in range(n_steps):
        idx = rng.choice(n, size=batch_size, replace=False)
        xb, yb = X_train[idx], y_train[idx]
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

    acc, loss = model_metrics(model, X_eval, y_eval)
    return {"model": model, "params": copy_param_dict(model), "acc": acc, "loss": loss}


def grad_direction(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], float]:
    plist = named_params(model)
    p = [t for _, t in plist]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X), y)
    g = torch.autograd.grad(loss, p, create_graph=False)
    gnorm = float(torch.sqrt(sum((x * x).sum() for x in g)).item())
    d = {n: (gg / max(gnorm, 1e-12)).detach().clone() for (n, _), gg in zip(plist, g)}
    return d, float(loss.item())


def random_direction(model: nn.Module, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
    d = {}
    sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        a = torch.from_numpy(rng.standard_normal(size=tuple(p.shape)).astype(np.float32))
        d[n] = a
        sq += float((a * a).sum().item())
    norm = math.sqrt(max(sq, 1e-12))
    return {k: v / norm for k, v in d.items()}


def compute_direction_rho_a(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    direction: Dict[str, torch.Tensor],
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = {n: p.detach() for n, p in model.named_parameters()}

    def f(pd):
        return functional_call(model, pd, (X,))

    z0 = f(base).detach()
    _, jvp_logits = jvp(f, (base,), (direction,))
    jvp_logits = jvp_logits.detach()

    da_t = jvp_logits.max(dim=1).values - jvp_logits.min(dim=1).values
    da = da_t.cpu().numpy()
    rho_a = math.pi / max(float(da_t.max().item()), 1e-12)

    z0_np = z0.cpu().numpy()
    jvp_np = jvp_logits.cpu().numpy()
    y_np = y.cpu().numpy()
    N = len(y_np)
    margins = np.zeros(N, dtype=np.float64)
    slopes = np.zeros(N, dtype=np.float64)
    for n in range(N):
        yc = int(y_np[n])
        others = z0_np[n].copy()
        others[yc] = -np.inf
        cc = int(np.argmax(others))
        margins[n] = z0_np[n, yc] - z0_np[n, cc]
        slopes[n] = jvp_np[n, cc] - jvp_np[n, yc]

    return rho_a, da, margins, slopes, jvp_np


def compute_directional_curvature(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    direction: Dict[str, torch.Tensor],
) -> Tuple[float, float, float]:
    plist = named_params(model)
    p = [t for _, t in plist]
    model.zero_grad(set_to_none=True)
    z = model(X)
    L0 = F.cross_entropy(z, y)

    g = torch.autograd.grad(L0, p, create_graph=True)
    dot = torch.zeros((), dtype=L0.dtype)
    for (n, _), gg in zip(plist, g):
        dot = dot + (gg * direction[n]).sum()
    hv = torch.autograd.grad(dot, p, create_graph=False, retain_graph=False)
    kappa = torch.zeros((), dtype=L0.dtype)
    for (n, _), h in zip(plist, hv):
        kappa = kappa + (h * direction[n]).sum()
    kappa_f = float(kappa.item())

    # Only positive curvature gives a finite local quadratic stability boundary.
    hess_boundary = 2.0 / kappa_f if kappa_f > 1e-12 else float("inf")
    return kappa_f, hess_boundary, float(L0.item())


def sweep_direction(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    direction: Dict[str, torch.Tensor],
    rho_a: float,
    L0: float,
    r_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = {n: p.detach() for n, p in model.named_parameters()}
    with torch.no_grad():
        z_base = functional_call(model, base, (X,))
        base_preds = z_base.argmax(dim=1)

    loss_ratios = np.zeros(len(r_values), dtype=np.float64)
    acc_values = np.zeros(len(r_values), dtype=np.float64)
    flip_fracs = np.zeros(len(r_values), dtype=np.float64)

    for i, r in enumerate(r_values):
        tau = float(r * rho_a)
        shifted = {n: base[n] + tau * direction[n] for n in base}
        with torch.no_grad():
            z = functional_call(model, shifted, (X,))
            L = float(F.cross_entropy(z, y).item())
            p = z.argmax(dim=1)
            loss_ratios[i] = L / max(L0, 1e-12)
            acc_values[i] = float((p == y).float().mean().item())
            flip_fracs[i] = float((p != base_preds).float().mean().item())

    return loss_ratios, acc_values, flip_fracs


def main() -> None:
    X, y = load_digits_csv()
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_ratio=0.3, seed=42)
    D, C = X_tr.shape[1], 10
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    # Different architecture from original MLP script.
    stages_cfg = [("early", 30), ("mid", 180), ("late", 700)]
    stages = {}
    for name, n_steps in stages_cfg:
        st = train_stage_linear(
            D,
            C,
            X_tr_t,
            y_tr_t,
            X_tr_t,
            y_tr_t,
            n_steps=n_steps,
            lr=0.1,
            batch_size=64,
            seed=SEED,
        )
        stages[name] = st
        print(f"Stage {name}: acc={st['acc']:.1%}, loss={st['loss']:.4f}")

    N_DIRS = 20
    print("\n" + "=" * 80)
    print("SWEEPING RANDOM DIRECTIONS: r = tau/rho_a")
    print("=" * 80)

    all_results = {}
    for stage_name, stage_info in stages.items():
        model = LinearNet(D, C)
        model.load_state_dict(stage_info["model"].state_dict())
        base_loss = stage_info["loss"]
        base_acc = stage_info["acc"]
        print(f"\n--- Stage: {stage_name} (acc={base_acc:.1%}, loss={base_loss:.4f}) ---")

        g_dir, _ = grad_direction(model, X_tr_t, y_tr_t)
        directions = [g_dir]
        labels = ["gradient"]
        rng = np.random.default_rng(SEED + len(stage_name))
        for i in range(N_DIRS):
            directions.append(random_direction(model, rng))
            labels.append(f"random_{i}")

        sr = {
            "rho_a": [],
            "hess_boundary": [],
            "kappa": [],
            "r_values": None,
            "loss_ratios": [],
            "acc_values": [],
            "flip_fracs": [],
            "is_gradient": [],
            "binding_margin": [],
            "binding_angle": [],
        }

        for di, (v, label) in enumerate(zip(directions, labels)):
            rho_a, da, margins, _slopes, _jvp = compute_direction_rho_a(model, X_tr_t, y_tr_t, v)
            kappa, hess_boundary, L0 = compute_directional_curvature(model, X_tr_t, y_tr_t, v)
            hess_r = hess_boundary / rho_a if np.isfinite(hess_boundary) else np.inf

            bind_idx = int(np.argmax(da))
            bind_margin = float(margins[bind_idx])
            bind_angle = float(np.degrees(np.arctan2(np.pi, abs(bind_margin))))

            r_max = max(15.0, 10.0 * hess_r) if np.isfinite(hess_r) else 35.0
            r_max = min(r_max, 100.0)
            r_small = np.logspace(-3, 0, 80)
            r_large = np.linspace(1.05, r_max, 120)
            r_values = np.concatenate([r_small, r_large])

            lr, accv, flips = sweep_direction(model, X_tr_t, y_tr_t, v, rho_a, L0, r_values)

            sr["rho_a"].append(rho_a)
            sr["hess_boundary"].append(hess_boundary)
            sr["kappa"].append(kappa)
            sr["loss_ratios"].append(lr)
            sr["acc_values"].append(accv)
            sr["flip_fracs"].append(flips)
            sr["is_gradient"].append(label == "gradient")
            sr["binding_margin"].append(bind_margin)
            sr["binding_angle"].append(bind_angle)
            sr["r_values"] = r_values

            if di == 0 or di % 5 == 0:
                htxt = f"{hess_r:.1f}x" if np.isfinite(hess_r) else "inf"
                print(
                    f"  Dir {di:2d} ({label:>10s}): rho_a={rho_a:.4f}, "
                    f"hess_bdy={hess_boundary:.4f} (={htxt}rho_a), "
                    f"kappa={kappa:.4f}, binding_angle={bind_angle:.0f}deg"
                )

        all_results[stage_name] = sr

    # Figure 1: 3x3 overview.
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    for si, (stage_name, sr) in enumerate(all_results.items()):
        r_vals = sr["r_values"]
        n_dirs = len(sr["rho_a"])
        is_grad = np.array(sr["is_gradient"], dtype=bool)

        # Loss ratio panel
        ax = axes[si, 0]
        for di in range(n_dirs):
            if is_grad[di]:
                continue
            ax.plot(r_vals, np.clip(sr["loss_ratios"][di], 0, 20), color="gray", alpha=0.15, lw=0.8)
        grad_idx = int(np.where(is_grad)[0][0])
        ax.plot(r_vals, np.clip(sr["loss_ratios"][grad_idx], 0, 20), color="red", lw=2.5, label="gradient dir")
        rnd_lr = np.array([sr["loss_ratios"][di] for di in range(n_dirs) if not is_grad[di]])
        med_lr = np.clip(np.median(rnd_lr, axis=0), 0, 20)
        q25_lr = np.clip(np.percentile(rnd_lr, 25, axis=0), 0, 20)
        q75_lr = np.clip(np.percentile(rnd_lr, 75, axis=0), 0, 20)
        ax.plot(r_vals, med_lr, color="blue", lw=2.5, label="median random")
        ax.fill_between(r_vals, q25_lr, q75_lr, alpha=0.15, color="blue")
        ax.axvline(1.0, color="red", lw=2, ls="--", alpha=0.7, label="r=1 (rho_a)")
        hess_r_vals = [
            sr["hess_boundary"][di] / sr["rho_a"][di]
            for di in range(n_dirs)
            if (not is_grad[di]) and np.isfinite(sr["hess_boundary"][di])
        ]
        med_hess_r = float(np.median(hess_r_vals)) if len(hess_r_vals) else np.nan
        if np.isfinite(med_hess_r):
            ax.axvline(
                med_hess_r,
                color="green",
                lw=2,
                ls=":",
                alpha=0.7,
                label=f"Hessian 2/k (med={med_hess_r:.1f}xrho_a)",
            )
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, min(15, float(r_vals.max())))
        ax.set_ylim(0, 10)
        ax.set_ylabel("Loss ratio L(theta+tau v)/L(theta)")
        if si == 0:
            ax.set_title("LOSS RATIO vs r = tau/rho_a", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2, which="both")
        ax.text(
            0.98,
            0.95,
            f"{stage_name}\nacc={stages[stage_name]['acc']:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Flip fraction panel
        ax = axes[si, 1]
        for di in range(n_dirs):
            if is_grad[di]:
                continue
            ax.plot(r_vals, sr["flip_fracs"][di], color="gray", alpha=0.15, lw=0.8)
        ax.plot(r_vals, sr["flip_fracs"][grad_idx], color="red", lw=2.5, label="gradient")
        rnd_ff = np.array([sr["flip_fracs"][di] for di in range(n_dirs) if not is_grad[di]])
        med_ff = np.median(rnd_ff, axis=0)
        q25_ff = np.percentile(rnd_ff, 25, axis=0)
        q75_ff = np.percentile(rnd_ff, 75, axis=0)
        ax.plot(r_vals, med_ff, color="blue", lw=2.5, label="median random")
        ax.fill_between(r_vals, q25_ff, q75_ff, alpha=0.15, color="blue")
        ax.axvline(1.0, color="red", lw=2, ls="--", alpha=0.7)
        if np.isfinite(med_hess_r):
            ax.axvline(med_hess_r, color="green", lw=2, ls=":", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, min(15, float(r_vals.max())))
        ax.set_ylabel("Fraction of classifications flipped")
        if si == 0:
            ax.set_title("CLASSIFICATION FLIPS vs r", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2, which="both")

        # Accuracy panel
        ax = axes[si, 2]
        for di in range(n_dirs):
            if is_grad[di]:
                continue
            ax.plot(r_vals, sr["acc_values"][di], color="gray", alpha=0.15, lw=0.8)
        ax.plot(r_vals, sr["acc_values"][grad_idx], color="red", lw=2.5, label="gradient")
        rnd_acc = np.array([sr["acc_values"][di] for di in range(n_dirs) if not is_grad[di]])
        med_acc = np.median(rnd_acc, axis=0)
        q25_acc = np.percentile(rnd_acc, 25, axis=0)
        q75_acc = np.percentile(rnd_acc, 75, axis=0)
        ax.plot(r_vals, med_acc, color="blue", lw=2.5, label="median random")
        ax.fill_between(r_vals, q25_acc, q75_acc, alpha=0.15, color="blue")
        ax.axvline(1.0, color="red", lw=2, ls="--", alpha=0.7)
        if np.isfinite(med_hess_r):
            ax.axvline(med_hess_r, color="green", lw=2, ls=":", alpha=0.7)
        ax.axhline(stages[stage_name]["acc"], color="black", lw=0.5, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, min(15, float(r_vals.max())))
        ax.set_ylabel("Accuracy")
        if si == 0:
            ax.set_title("ACCURACY vs r", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2, which="both")

    for ax in axes[2]:
        ax.set_xlabel("r = tau / rho_a")

    plt.suptitle(
        "RANDOM DIRECTIONS (Fixed): Is r = tau/rho_a = 1 Universal?\n"
        "Gray = random dirs, Red = gradient, Blue = median random\n"
        "Red dashed = rho_a boundary, Green dotted = Hessian boundary (positive-curvature dirs)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    f1 = FIG_DIR / "exp11_random_direction_sweep_fixed.png"
    plt.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved: {f1}]")

    # Figure 2: log-scale near r=1.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for si, (stage_name, sr) in enumerate(all_results.items()):
        ax = axes[si]
        r_vals = sr["r_values"]
        n_dirs = len(sr["rho_a"])
        is_grad = np.array(sr["is_gradient"], dtype=bool)
        rnd_lr = np.array([sr["loss_ratios"][di] for di in range(n_dirs) if not is_grad[di]])
        for row in rnd_lr:
            ax.plot(r_vals, np.clip(row, 0.01, 100), color="gray", alpha=0.12, lw=0.7)
        grad_idx = int(np.where(is_grad)[0][0])
        ax.plot(r_vals, np.clip(sr["loss_ratios"][grad_idx], 0.01, 100), color="red", lw=2.5, label="gradient")
        med = np.median(rnd_lr, axis=0)
        ax.plot(r_vals, np.clip(med, 0.01, 100), color="blue", lw=2.5, label="median random")
        ax.axvline(1.0, color="red", lw=2.5, ls="--", alpha=0.8, label="r=1")
        hess_r_vals = [
            sr["hess_boundary"][di] / sr["rho_a"][di]
            for di in range(n_dirs)
            if (not is_grad[di]) and np.isfinite(sr["hess_boundary"][di])
        ]
        med_hess_r = float(np.median(hess_r_vals)) if len(hess_r_vals) else np.nan
        if np.isfinite(med_hess_r):
            ax.axvline(med_hess_r, color="green", lw=2.5, ls=":", alpha=0.8, label=f"2/k = {med_hess_r:.1f}xrho_a")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, min(50, float(r_vals.max())))
        ax.set_ylim(0.1, 50)
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.3)
        ax.set_xlabel("r = tau / rho_a")
        ax.set_ylabel("Loss ratio")
        ax.set_title(f"{stage_name} (acc={stages[stage_name]['acc']:.0%})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, which="both")

    plt.suptitle(
        "LOG-SCALE: Loss Inflation Onset near r = 1 (Random Directions)\n"
        "Does transition align better with rho_a (red) than Hessian 2/k (green)?",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    f2 = FIG_DIR / "exp11_random_direction_logscale_fixed.png"
    plt.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved: {f2}]")

    # Figure 3: rho_a vs Hessian scatter.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for si, (stage_name, sr) in enumerate(all_results.items()):
        ax = axes[si]
        rho_vals = np.array(sr["rho_a"])
        hess_vals = np.array(sr["hess_boundary"])
        is_grad = np.array(sr["is_gradient"], dtype=bool)
        rnd_mask = (~is_grad) & np.isfinite(hess_vals)
        ax.scatter(rho_vals[rnd_mask], hess_vals[rnd_mask], c="blue", alpha=0.5, s=50, label="random dirs", zorder=5)
        grad_mask = is_grad & np.isfinite(hess_vals)
        if np.any(grad_mask):
            ax.scatter(rho_vals[grad_mask], hess_vals[grad_mask], c="red", s=200, marker="*", label="gradient dir", zorder=10)
        lim = max(float(np.nanmax(rho_vals)), float(np.nanmax(hess_vals[np.isfinite(hess_vals)]))) * 1.2
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="rho_a = 2/k")
        binding = float(np.mean(rho_vals[rnd_mask] < hess_vals[rnd_mask])) if np.any(rnd_mask) else np.nan
        ax.set_xlabel("rho_a (analyticity radius)")
        ax.set_ylabel("2/k (Hessian boundary)")
        ax.set_title(
            f"{stage_name} (acc={stages[stage_name]['acc']:.0%})\n"
            f"rho_a < 2/k in {binding:.0%} of random dirs",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    plt.suptitle(
        "rho_a vs Hessian Boundary (Fixed)\n"
        "Points above diagonal mean rho_a is the tighter (earlier) constraint",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    f3 = FIG_DIR / "exp11_rho_vs_hessian_scatter_fixed.png"
    plt.savefig(f3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved: {f3}]")

    # Figure 4: histogram of crossing points at 2x loss inflation.
    print("\n" + "=" * 80)
    print("WHERE DOES LOSS RATIO FIRST EXCEED 2x?")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for si, (stage_name, sr) in enumerate(all_results.items()):
        r_vals = sr["r_values"]
        is_grad = np.array(sr["is_gradient"], dtype=bool)
        cross_r_rho = []
        cross_r_hess = []
        for di in range(len(sr["rho_a"])):
            if is_grad[di]:
                continue
            lr = np.array(sr["loss_ratios"][di])
            rho = sr["rho_a"][di]
            hess = sr["hess_boundary"][di]
            idx = np.where(lr > 2.0)[0]
            if len(idx) == 0:
                continue
            r_cross = float(r_vals[idx[0]])
            tau_cross = r_cross * rho
            cross_r_rho.append(r_cross)
            if np.isfinite(hess) and hess > 1e-12:
                cross_r_hess.append(tau_cross / hess)

        cross_r_rho = np.array(cross_r_rho, dtype=np.float64)
        cross_r_hess = np.array(cross_r_hess, dtype=np.float64)

        # Top row (rho units)
        ax = axes[0, si]
        if len(cross_r_rho):
            ax.hist(cross_r_rho, bins=15, color="blue", alpha=0.7, edgecolor="black")
            ax.axvline(1.0, color="red", lw=2.5, ls="--", label="r=1")
            ax.axvline(np.median(cross_r_rho), color="blue", lw=2, label=f"median={np.median(cross_r_rho):.2f}")
        ax.set_title(f"{stage_name} (acc={stages[stage_name]['acc']:.0%})", fontsize=13, fontweight="bold")
        ax.set_xlabel("r at 2x loss inflation (rho_a units)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Bottom row (hessian units)
        ax = axes[1, si]
        if len(cross_r_hess):
            ax.hist(cross_r_hess, bins=15, color="green", alpha=0.7, edgecolor="black")
            ax.axvline(1.0, color="red", lw=2.5, ls="--", label="r=1")
            ax.axvline(np.median(cross_r_hess), color="green", lw=2, label=f"median={np.median(cross_r_hess):.2f}")
        ax.set_xlabel("r at 2x loss inflation (Hessian units)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        print(f"\n--- {stage_name} (acc={stages[stage_name]['acc']:.0%}) ---")
        if len(cross_r_rho):
            print(
                f"  rho_a units: median={np.median(cross_r_rho):.3f}, "
                f"IQR=[{np.percentile(cross_r_rho,25):.3f}, {np.percentile(cross_r_rho,75):.3f}]"
            )
        if len(cross_r_hess):
            print(
                f"  Hessian units: median={np.median(cross_r_hess):.3f}, "
                f"IQR=[{np.percentile(cross_r_hess,25):.3f}, {np.percentile(cross_r_hess,75):.3f}]"
            )
        if len(cross_r_rho) > 2 and len(cross_r_hess) > 2:
            cv_rho = np.std(cross_r_rho) / max(np.mean(cross_r_rho), 1e-12)
            cv_h = np.std(cross_r_hess) / max(np.mean(cross_r_hess), 1e-12)
            better = "rho_a" if cv_rho < cv_h else "Hessian"
            print(f"  CV(r_rho)={cv_rho:.3f}, CV(r_hess)={cv_h:.3f} -> {better} tighter clustering")

    axes[0, 0].set_ylabel("Count\n(rho_a units)")
    axes[1, 0].set_ylabel("Count\n(Hessian units)")
    plt.suptitle(
        "Crossing Histograms (2x loss inflation)\n"
        "Top: rho_a-normalized, Bottom: Hessian-normalized",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    f4 = FIG_DIR / "exp11_crossing_point_histograms_fixed.png"
    plt.savefig(f4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved: {f4}]")

    # Save compact results.
    npz = {"stage_names": np.array(list(all_results.keys()), dtype=object)}
    for stage_name, sr in all_results.items():
        npz[f"{stage_name}_r_values"] = sr["r_values"]
        npz[f"{stage_name}_rho_a"] = np.array(sr["rho_a"], dtype=np.float64)
        npz[f"{stage_name}_hess_boundary"] = np.array(sr["hess_boundary"], dtype=np.float64)
        npz[f"{stage_name}_kappa"] = np.array(sr["kappa"], dtype=np.float64)
        npz[f"{stage_name}_is_gradient"] = np.array(sr["is_gradient"], dtype=np.int32)
        npz[f"{stage_name}_loss_ratios"] = np.array(sr["loss_ratios"], dtype=np.float64)
        npz[f"{stage_name}_acc_values"] = np.array(sr["acc_values"], dtype=np.float64)
        npz[f"{stage_name}_flip_fracs"] = np.array(sr["flip_fracs"], dtype=np.float64)
    npz_path = FIG_DIR / "exp11_random_direction_sweep_fixed_results.npz"
    np.savez(npz_path, **npz)
    print(f"[Saved: {npz_path}]")


if __name__ == "__main__":
    main()
