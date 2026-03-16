#!/usr/bin/env python3
"""Multi-seed architecture comparison with LR spike (Figure 8).

Compares Transformer, MLP+LN, CNN+BN with 10x spike at step 50.
Plain Adam vs rho-controller.

Outputs:
  - cache/archgrid_multiseed.pt
  - paper/figures/plots/archgrid-jvp-twocol.{png,pdf}
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import site
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-exp26")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "cache"
PLOT_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ghosts.reporting import repo_relpath, scalar_stats, write_summary

PALETTE = {
    'red': '#E3120B',
    'gold': '#F4A100',
    'blue': '#1E5AA8',
    'teal': '#0097A7',
    'dark': '#3D3D3D',
    'mid': '#767676',
    'light': '#D0D0D0',
}


def load_digits() -> Tuple[np.ndarray, np.ndarray]:
    candidates: List[Path] = []
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


# === ARCHITECTURES ===

class Transformer(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(64, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.ffn1 = nn.Linear(dim, dim * 4)
        self.ffn2 = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).unsqueeze(1)
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        x = x + self.out(torch.matmul(F.softmax(a, -1), v))
        x = x + self.ffn2(F.gelu(self.ffn1(self.norm2(x))))
        return self.head(x.squeeze(1))


class MLPLN(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc3 = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.fc1(x)))
        x = F.gelu(self.norm2(self.fc2(x)))
        x = F.gelu(self.norm3(self.fc3(x)))
        return self.head(x)


class CNNBN(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


ARCH_CLASSES = {
    "Transformer": Transformer,
    "MLP+LN": MLPLN,
    "CNN+BN": CNNBN,
}


def make_model(arch: str, device: torch.device) -> nn.Module:
    return ARCH_CLASSES[arch]().to(device)


def compute_rhoA_jvp(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                     direction: torch.Tensor) -> float:
    """Compute rho_a = pi / Delta_a via JVP."""
    # Put model in eval mode to avoid BatchNorm running stats mutation
    was_training = model.training
    model.eval()

    params = dict(model.named_parameters())
    param_shapes = {k: p.shape for k, p in params.items()}

    # Build tangent dict
    tangents = {}
    offset = 0
    for k, shape in param_shapes.items():
        size = int(np.prod(shape))
        tangents[k] = direction[offset:offset + size].view(shape)
        offset += size

    def fwd(p):
        return functional_call(model, p, (X,))

    _, dlogits = jvp(fwd, (params,), (tangents,))
    spread = dlogits.max(dim=1).values - dlogits.min(dim=1).values
    delta_a = float(spread.max().item())

    # Restore training mode
    if was_training:
        model.train()

    return math.pi / max(delta_a, 1e-12)


def train_step_plain(model: nn.Module, opt: torch.optim.Optimizer,
                     X: torch.Tensor, y: torch.Tensor, lr: float) -> Dict:
    """Plain Adam step."""
    for pg in opt.param_groups:
        pg['lr'] = lr
    opt.zero_grad(set_to_none=True)
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()

    with torch.no_grad():
        acc = (logits.argmax(1) == y).float().mean().item()

    return {"loss": loss.item(), "acc": acc}


def train_step_clip(model: nn.Module, opt: torch.optim.Optimizer,
                    X: torch.Tensor, y: torch.Tensor, lr: float,
                    max_norm: float = 1.0) -> Dict:
    """Adam + gradient clipping step."""
    for pg in opt.param_groups:
        pg['lr'] = lr
    opt.zero_grad(set_to_none=True)
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    opt.step()

    with torch.no_grad():
        acc = (logits.argmax(1) == y).float().mean().item()

    return {"loss": loss.item(), "acc": acc}


def train_step_rho(model: nn.Module, opt: torch.optim.Optimizer,
                   X: torch.Tensor, y: torch.Tensor, lr: float) -> Dict:
    """Rho-controller step."""
    opt.zero_grad(set_to_none=True)
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Get gradient direction
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    g = torch.cat(grads)
    g_norm = g.norm().item()
    if g_norm < 1e-12:
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss.item(), "acc": acc, "rhoA": float('inf'), "tau": 0.0}

    v = -g / g.norm()
    rhoA = compute_rhoA_jvp(model, X, y, v)

    # Compute tau and clip
    tau_raw = lr * g_norm
    tau = min(tau_raw, rhoA)
    scale = tau / tau_raw if tau_raw > 0 else 1.0

    # Apply scaled update
    for pg in opt.param_groups:
        pg['lr'] = lr * scale
    opt.step()

    with torch.no_grad():
        acc = (logits.argmax(1) == y).float().mean().item()

    return {"loss": loss.item(), "acc": acc, "rhoA": rhoA, "tau": tau}


def run_single(arch: str, seed: int, steps: int, base_lr: float, spike_at: int,
               spike_mul: float, device: torch.device) -> Dict:
    """Run one seed for one architecture: plain and rho_ctrl."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_np, y_np = load_digits()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    # Split for test accuracy
    n_test = int(0.2 * len(X))
    Xtr, Xte = X[n_test:], X[:n_test]
    ytr, yte = y[n_test:], y[:n_test]

    results = {"plain": {"loss": [], "acc": []},
               "grad_clip": {"loss": [], "acc": []},
               "rho_ctrl": {"loss": [], "acc": [], "rhoA": [], "tau": []}}

    # Plain Adam
    torch.manual_seed(seed)
    model_plain = make_model(arch, device)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=base_lr)

    for step in range(steps):
        lr = base_lr * spike_mul if step == spike_at else base_lr
        out = train_step_plain(model_plain, opt_plain, Xtr, ytr, lr)
        results["plain"]["loss"].append(out["loss"])
        with torch.no_grad():
            acc_te = (model_plain(Xte).argmax(1) == yte).float().mean().item()
        results["plain"]["acc"].append(acc_te)

    # Adam + gradient clipping (max_norm=1.0)
    torch.manual_seed(seed)
    model_clip = make_model(arch, device)
    opt_clip = torch.optim.Adam(model_clip.parameters(), lr=base_lr)

    for step in range(steps):
        lr = base_lr * spike_mul if step == spike_at else base_lr
        out = train_step_clip(model_clip, opt_clip, Xtr, ytr, lr, max_norm=1.0)
        results["grad_clip"]["loss"].append(out["loss"])
        with torch.no_grad():
            acc_te = (model_clip(Xte).argmax(1) == yte).float().mean().item()
        results["grad_clip"]["acc"].append(acc_te)

    # Rho-controller
    torch.manual_seed(seed)
    model_rho = make_model(arch, device)
    opt_rho = torch.optim.Adam(model_rho.parameters(), lr=base_lr)

    for step in range(steps):
        lr = base_lr * spike_mul if step == spike_at else base_lr
        out = train_step_rho(model_rho, opt_rho, Xtr, ytr, lr)
        results["rho_ctrl"]["loss"].append(out["loss"])
        results["rho_ctrl"]["rhoA"].append(out.get("rhoA", float('inf')))
        results["rho_ctrl"]["tau"].append(out.get("tau", 0.0))
        with torch.no_grad():
            acc_te = (model_rho(Xte).argmax(1) == yte).float().mean().item()
        results["rho_ctrl"]["acc"].append(acc_te)

    return results


def run_experiment(archs: List[str], seeds: List[int], steps: int, base_lr: float,
                   spike_at: int, spike_mul: float, device: torch.device,
                   print_every: int = 1) -> Dict:
    """Run all archs × seeds."""
    data = {arch: {} for arch in archs}
    for arch in archs:
        print(f"  {arch}")
        for i, seed in enumerate(seeds):
            if (i + 1) % print_every == 0 or i == 0:
                print(f"    seed {seed} ({i+1}/{len(seeds)})")
            data[arch][seed] = run_single(arch, seed, steps, base_lr, spike_at,
                                          spike_mul, device)
    return data


def make_plot(data: Dict, archs: List[str], steps: int, spike_at: int,
              out_png: Path, out_pdf: Path, spike_mul: float = 10.0) -> None:
    """Plot median+IQR for loss and accuracy per architecture."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    def stats(arrs):
        arr = np.array(arrs)
        return np.median(arr, axis=0), np.percentile(arr, 25, axis=0), \
               np.percentile(arr, 75, axis=0)

    fig = plt.figure(figsize=(8, 7.5))
    gs = gridspec.GridSpec(6, 2, height_ratios=[0.18, 1, 0.18, 1, 0.18, 1],
                           hspace=0.12, wspace=0.30)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.11, right=0.96)

    fig.suptitle(f'Architecture Comparison: {int(spike_mul)}x LR Spike (Multi-seed)',
                 fontsize=13, fontweight='bold', y=0.97)

    # Legend
    fig.text(0.18, 0.92, '- Plain', fontsize=10, color=PALETTE['red'],
             fontweight='bold', ha='left')
    fig.text(0.38, 0.92, '- Clip', fontsize=10, color=PALETTE['teal'],
             fontweight='bold', ha='left')
    fig.text(0.58, 0.92, r'- $\rho_a$ ctrl', fontsize=10, color=PALETTE['gold'],
             fontweight='bold', ha='left')

    # Column titles
    fig.text(0.33, 0.885, 'Training Loss', fontsize=11, fontweight='bold',
             ha='center', color=PALETTE['dark'])
    fig.text(0.74, 0.885, 'Test Accuracy', fontsize=11, fontweight='bold',
             ha='center', color=PALETTE['dark'])

    MESSAGES = {
        "Transformer": "Attention amplifies spike",
        "MLP+LN": "LayerNorm dampens but insufficient",
        "CNN+BN": "BatchNorm causes collapse",
    }

    x = np.arange(steps)

    for i, arch in enumerate(archs):
        textrow = i * 2
        plotrow = i * 2 + 1

        # Text row
        ax_text = fig.add_subplot(gs[textrow, :])
        ax_text.set_axis_off()
        msg = MESSAGES.get(arch, "")
        ax_text.text(0.5, 0.5, f"{arch}: {msg}", transform=ax_text.transAxes,
                     fontsize=11, fontweight='bold', color=PALETTE['blue'],
                     ha='center', va='center')

        # Gather data
        seeds = list(data[arch].keys())
        plain_loss = [data[arch][s]["plain"]["loss"] for s in seeds]
        plain_acc = [data[arch][s]["plain"]["acc"] for s in seeds]
        # Check if grad_clip exists (backward compat)
        has_clip = "grad_clip" in data[arch][seeds[0]]
        if has_clip:
            clip_loss = [data[arch][s]["grad_clip"]["loss"] for s in seeds]
            clip_acc = [data[arch][s]["grad_clip"]["acc"] for s in seeds]
        rho_loss = [data[arch][s]["rho_ctrl"]["loss"] for s in seeds]
        rho_acc = [data[arch][s]["rho_ctrl"]["acc"] for s in seeds]

        # Column 0: Loss
        ax = fig.add_subplot(gs[plotrow, 0])
        med_p, q25_p, q75_p = stats(plain_loss)
        med_r, q25_r, q75_r = stats(rho_loss)

        ax.fill_between(x, np.maximum(q25_p, 1e-6), np.maximum(q75_p, 1e-6),
                        alpha=0.15, color=PALETTE['red'])
        ax.semilogy(x, np.maximum(med_p, 1e-6), color=PALETTE['red'], lw=2)
        if has_clip:
            med_c, q25_c, q75_c = stats(clip_loss)
            ax.fill_between(x, np.maximum(q25_c, 1e-6), np.maximum(q75_c, 1e-6),
                            alpha=0.15, color=PALETTE['teal'])
            ax.semilogy(x, np.maximum(med_c, 1e-6), color=PALETTE['teal'], lw=2)
        ax.fill_between(x, np.maximum(q25_r, 1e-6), np.maximum(q75_r, 1e-6),
                        alpha=0.15, color=PALETTE['gold'])
        ax.semilogy(x, np.maximum(med_r, 1e-6), color=PALETTE['gold'], lw=2)
        ax.axvline(spike_at, color=PALETTE['mid'], ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0, steps)
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE['light'])

        if i == 0:
            ax.annotate(f'{int(spike_mul)}x spike', xy=(spike_at, 0.6),
                        xycoords=('data', 'axes fraction'),
                        xytext=(15, 5), textcoords='offset points',
                        fontsize=9, color=PALETTE['dark'],
                        arrowprops=dict(arrowstyle='->', color=PALETTE['mid']))
        if i == len(archs) - 1:
            ax.set_xlabel('Step', fontsize=11)
        else:
            ax.set_xticklabels([])

        # Column 1: Accuracy
        ax = fig.add_subplot(gs[plotrow, 1])
        med_p, q25_p, q75_p = stats(plain_acc)
        med_r, q25_r, q75_r = stats(rho_acc)

        ax.fill_between(x, np.clip(q25_p, 0, 1), np.clip(q75_p, 0, 1),
                        alpha=0.15, color=PALETTE['red'])
        ax.plot(x, med_p, color=PALETTE['red'], lw=2)
        if has_clip:
            med_c, q25_c, q75_c = stats(clip_acc)
            ax.fill_between(x, np.clip(q25_c, 0, 1), np.clip(q75_c, 0, 1),
                            alpha=0.15, color=PALETTE['teal'])
            ax.plot(x, med_c, color=PALETTE['teal'], lw=2)
        ax.fill_between(x, np.clip(q25_r, 0, 1), np.clip(q75_r, 0, 1),
                        alpha=0.15, color=PALETTE['gold'])
        ax.plot(x, med_r, color=PALETTE['gold'], lw=2)
        ax.axvline(spike_at, color=PALETTE['mid'], ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0, steps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0%', '50%', '100%'])
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE['light'])

        if i == len(archs) - 1:
            ax.set_xlabel('Step', fontsize=11)
        else:
            ax.set_xticklabels([])

    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def build_summary(data: Dict, archs: List[str], config: Dict,
                  out_pt: Path, out_png: Path, out_pdf: Path,
                  out_summary: Path) -> Dict:
    summary = {
        "experiment_id": "archgrid",
        "config": config,
        "artifacts": {
            "data": repo_relpath(out_pt, REPO_ROOT),
            "plot_png": repo_relpath(out_png, REPO_ROOT),
            "plot_pdf": repo_relpath(out_pdf, REPO_ROOT),
            "summary_json": repo_relpath(out_summary, REPO_ROOT),
        },
        "architectures": {},
    }
    spike_at = int(config["spike_at"])
    for arch in archs:
        seeds = sorted(data[arch].keys())
        arch_summary = {}
        for mode in ["plain", "grad_clip", "rho_ctrl"]:
            if mode not in data[arch][seeds[0]]:
                continue
            final_loss = [data[arch][seed][mode]["loss"][-1] for seed in seeds]
            final_acc = [data[arch][seed][mode]["acc"][-1] for seed in seeds]
            peak_loss = [
                float(np.max(np.asarray(data[arch][seed][mode]["loss"][spike_at:], dtype=float)))
                for seed in seeds
            ]
            mode_summary = {
                "final_loss": scalar_stats(final_loss),
                "final_acc": scalar_stats(final_acc),
                "peak_loss_after_spike": scalar_stats(peak_loss),
            }
            if mode == "rho_ctrl":
                max_r = []
                for seed in seeds:
                    tau = np.asarray(data[arch][seed][mode]["tau"], dtype=float)
                    rho = np.asarray(data[arch][seed][mode]["rhoA"], dtype=float)
                    valid = np.isfinite(rho) & (rho > 0)
                    ratios = np.full_like(tau, np.nan, dtype=float)
                    ratios[valid] = tau[valid] / rho[valid]
                    max_r.append(float(np.nanmax(ratios)))
                mode_summary["max_tau_over_rho"] = scalar_stats(max_r)
            arch_summary[mode] = mode_summary
        summary["architectures"][arch] = arch_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed archgrid test.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--archs", type=str, default="Transformer,MLP+LN,CNN+BN")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--base-lr", type=float, default=0.015)
    parser.add_argument("--spike-at", type=int, default=50)
    parser.add_argument("--spike-mul", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_pt = DATA_DIR / f"archgrid_multiseed{tag}.pt"
    out_png = PLOT_DIR / f"archgrid-jvp-twocol{tag}.png"
    out_pdf = PLOT_DIR / f"archgrid-jvp-twocol{tag}.pdf"
    out_summary = PLOT_DIR / f"summary{tag}.json"

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]

    if args.replot:
        if not out_pt.exists():
            print(f"Error: data file not found: {out_pt}")
            return
        payload = torch.load(out_pt, weights_only=False)
        data = payload["data"]
        archs = payload["config"]["archs"]
        steps = payload["config"]["steps"]
        spike_at = payload["config"]["spike_at"]
        print(f"Replotting from {out_pt}")
        spike_mul = payload["config"]["spike_mul"]
        make_plot(data, archs, steps, spike_at, out_png, out_pdf, spike_mul)
        print(f"saved: {out_png}")
        print(f"saved: {out_pdf}")
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device} seeds={seeds} archs={archs}")
    print(f"steps={args.steps} base_lr={args.base_lr} spike_at={args.spike_at} "
          f"spike_mul={args.spike_mul}")
    print("Running architecture comparison...")
    data = run_experiment(archs, seeds, args.steps, args.base_lr, args.spike_at,
                          args.spike_mul, device)

    payload = {
        "config": {
            "seeds": seeds, "archs": archs, "steps": args.steps,
            "base_lr": args.base_lr, "spike_at": args.spike_at,
            "spike_mul": args.spike_mul,
        },
        "data": data,
    }
    torch.save(payload, out_pt)
    make_plot(data, archs, args.steps, args.spike_at, out_png, out_pdf,
              args.spike_mul)
    write_summary(
        out_summary,
        build_summary(payload["data"], archs, payload["config"], out_pt, out_png, out_pdf, out_summary),
    )

    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
