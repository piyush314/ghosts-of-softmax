#!/usr/bin/env python3
"""Multi-seed temperature-scaling fingerprint test (Figure 7).

Trains MLP on digits at different temperatures, sweeps one-step sizes,
verifies curves collapse when scaled by r_T = tau * Delta_a / (pi * T).

Outputs:
  - cache/fingerprint_multiseed.pt
  - paper/figures/plots/theory-fingerprint-tests_v4.{png,pdf}
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-exp25")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
    0.25: "#7B1FA2",
    0.5: "#2E7D32",
    1.0: "#006BA2",
    2.0: "#F4A100",
    4.0: "#D84315",
    8.0: "#5D4037",
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


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(test_ratio * len(idx)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    tr, te = np.concatenate(train_idx), np.concatenate(test_idx)
    rng.shuffle(tr)
    rng.shuffle(te)
    return X[tr], X[te], y[tr], y[te]


class MLPDigits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.head(x)


def ce_temp(logits: torch.Tensor, y: torch.Tensor, temp: float) -> torch.Tensor:
    return F.cross_entropy(logits / temp, y)


def train_model(temp: float, seed: int, steps: int, lr: float, batch: int,
                device: torch.device) -> Tuple[MLPDigits, torch.Tensor, torch.Tensor]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    X, y = load_digits()
    Xtr, _, ytr, _ = stratified_split(X, y, test_ratio=0.2, seed=seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)

    model = MLPDigits().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 17)
    n = len(Xtr)
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        opt.zero_grad(set_to_none=True)
        loss = ce_temp(model(Xtr[idx]), ytr[idx], temp)
        loss.backward()
        opt.step()
    return model, Xtr, ytr


def grad_direction(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                   temp: float) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = ce_temp(model(X), y, temp)
    loss.backward()
    grads = []
    for p in model.parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grads.append(g.detach().flatten())
    gvec = torch.cat(grads)
    return -gvec / (gvec.norm() + 1e-12)


@torch.no_grad()
def finite_diff_dlogits(model: nn.Module, X: torch.Tensor, u: torch.Tensor,
                        eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    base = parameters_to_vector(model.parameters()).detach().clone()
    logits0 = model(X).detach()
    vector_to_parameters(base + eps * u, model.parameters())
    logits1 = model(X).detach()
    vector_to_parameters(base, model.parameters())
    return logits0, (logits1 - logits0) / eps


@torch.no_grad()
def one_step_curve(model: nn.Module, X: torch.Tensor, y: torch.Tensor, temp: float,
                   u: torch.Tensor, tau_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base = parameters_to_vector(model.parameters()).detach().clone()
    logits_base = model(X).detach()
    pred_base = logits_base.argmax(1)
    loss0 = float(ce_temp(logits_base, y, temp).item())

    loss_ratio = np.zeros_like(tau_grid, dtype=float)
    flip_frac = np.zeros_like(tau_grid, dtype=float)
    for i, tau in enumerate(tau_grid):
        vector_to_parameters(base + float(tau) * u, model.parameters())
        logits = model(X).detach()
        loss_ratio[i] = float(ce_temp(logits, y, temp).item()) / max(loss0, 1e-12)
        flip_frac[i] = float((logits.argmax(1) != pred_base).float().mean().item())
    vector_to_parameters(base, model.parameters())
    return loss_ratio, flip_frac


def run_single_seed(temps: List[float], seed: int, train_steps: int, train_lr: float,
                    batch_size: int, device: torch.device) -> Dict[float, Dict[str, np.ndarray]]:
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for t in temps:
        model, Xtr, ytr = train_model(t, seed, train_steps, train_lr, batch_size, device)
        u = grad_direction(model, Xtr, ytr, t)
        _, dlogits = finite_diff_dlogits(model, Xtr, u, eps=1e-4)
        spread = dlogits.max(1).values - dlogits.min(1).values
        delta_a = float(spread.max().item())
        rho_t = (math.pi * t) / max(delta_a, 1e-12)

        tau_max = max(4.0 * rho_t, 2.0)
        tau_grid = np.geomspace(1e-4, tau_max, 80)
        loss_ratio, flip_frac = one_step_curve(model, Xtr, ytr, t, u, tau_grid)
        r_scaled = tau_grid * delta_a / (math.pi * t)

        out[t] = {
            "tau": tau_grid,
            "r_scaled": r_scaled,
            "loss_ratio": loss_ratio,
            "flip_frac": flip_frac,
            "delta_a": delta_a,
            "rho_t": rho_t,
        }
    return out


def run_experiment(seeds: List[int], temps: List[float], train_steps: int,
                   train_lr: float, batch_size: int, device: torch.device,
                   print_every: int = 1) -> Dict:
    all_data: Dict[int, Dict[float, Dict[str, np.ndarray]]] = {}
    for i, seed in enumerate(seeds):
        if (i + 1) % print_every == 0 or i == 0:
            print(f"  seed {seed} ({i+1}/{len(seeds)})")
        all_data[seed] = run_single_seed(temps, seed, train_steps, train_lr,
                                         batch_size, device)
    return all_data


def make_plot(data: Dict, temps: List[float], out_png: Path, out_pdf: Path) -> None:
    """Plot median+IQR for raw tau and scaled r_T."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Common r_scaled grid for interpolation
    r_grid = np.geomspace(1e-3, 10.0, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))
    fig.subplots_adjust(top=0.80, wspace=0.25)
    fig.suptitle("Temperature Fingerprint: Multi-seed Median+IQR", fontsize=13,
                 fontweight="bold")

    seeds = list(data.keys())

    # Panel A: Raw tau
    ax = axes[0]
    for t in temps:
        color = PALETTE.get(t, "#333333")
        # Gather curves from all seeds
        all_tau = []
        all_loss = []
        for seed in seeds:
            d = data[seed][t]
            all_tau.append(d["tau"])
            all_loss.append(d["loss_ratio"])

        # Find common tau range
        tau_min = max(arr.min() for arr in all_tau)
        tau_max = min(arr.max() for arr in all_tau)
        tau_grid = np.geomspace(tau_min, tau_max, 100)

        # Interpolate all seeds to common grid
        interp_loss = []
        for tau_arr, loss_arr in zip(all_tau, all_loss):
            interp_loss.append(np.interp(tau_grid, tau_arr, loss_arr))
        interp_loss = np.array(interp_loss)

        med = np.median(interp_loss, axis=0)
        q25 = np.percentile(interp_loss, 25, axis=0)
        q75 = np.percentile(interp_loss, 75, axis=0)

        ax.fill_between(tau_grid, q25, q75, alpha=0.2, color=color)
        ax.plot(tau_grid, med, color=color, lw=2, label=f"T={t:g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Raw step size $\tau$")
    ax.set_ylabel("One-step loss ratio")
    ax.set_title("A) Raw tau (curves spread)", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)

    # Panel B: Scaled r_T
    ax = axes[1]
    for t in temps:
        color = PALETTE.get(t, "#333333")
        # Interpolate to common r_grid
        interp_loss = []
        for seed in seeds:
            d = data[seed][t]
            r_arr = d["r_scaled"]
            loss_arr = d["loss_ratio"]
            # Only interpolate within valid range
            valid = (r_grid >= r_arr.min()) & (r_grid <= r_arr.max())
            loss_interp = np.full_like(r_grid, np.nan)
            loss_interp[valid] = np.interp(r_grid[valid], r_arr, loss_arr)
            interp_loss.append(loss_interp)
        interp_loss = np.array(interp_loss)

        med = np.nanmedian(interp_loss, axis=0)
        q25 = np.nanpercentile(interp_loss, 25, axis=0)
        q75 = np.nanpercentile(interp_loss, 75, axis=0)

        valid_mask = ~np.isnan(med)
        ax.fill_between(r_grid[valid_mask], q25[valid_mask], q75[valid_mask],
                        alpha=0.2, color=color)
        ax.plot(r_grid[valid_mask], med[valid_mask], color=color, lw=2,
                label=f"T={t:g}")

    ax.axvline(1.0, color="#888", ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Scaled step $r_T = \tau\Delta_a/(\pi T)$")
    ax.set_ylabel("One-step loss ratio")
    ax.set_title("B) Scaled r_T (curves collapse)", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    idx = np.where(np.asarray(y, dtype=float) >= threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float(np.asarray(x, dtype=float)[idx[0]])


def build_summary(data: Dict, temps: List[float], config: Dict,
                  out_pt: Path, out_png: Path, out_pdf: Path,
                  out_summary: Path) -> Dict:
    seeds = sorted(data.keys())
    summary = {
        "experiment_id": "tempfingerprint",
        "config": config,
        "artifacts": {
            "data": repo_relpath(out_pt, REPO_ROOT),
            "plot_png": repo_relpath(out_png, REPO_ROOT),
            "plot_pdf": repo_relpath(out_pdf, REPO_ROOT),
            "summary_json": repo_relpath(out_summary, REPO_ROOT),
        },
        "temperatures": {},
    }
    for temp in temps:
        delta_a = []
        rho_t = []
        loss2_tau = []
        loss2_r = []
        for seed in seeds:
            item = data[seed][temp]
            delta_a.append(float(item["delta_a"]))
            rho_t.append(float(item["rho_t"]))
            loss2_tau.append(first_crossing(item["tau"], item["loss_ratio"], 2.0))
            loss2_r.append(first_crossing(item["r_scaled"], item["loss_ratio"], 2.0))
        summary["temperatures"][str(temp)] = {
            "delta_a": scalar_stats(delta_a),
            "rho_t": scalar_stats(rho_t),
            "loss2_cross_tau": scalar_stats(loss2_tau),
            "loss2_cross_r_scaled": scalar_stats(loss2_r),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed fingerprint test.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--temps", type=str, default="0.25,0.5,1,2,4,8")
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--train-lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_pt = DATA_DIR / f"fingerprint_multiseed{tag}.pt"
    out_png = PLOT_DIR / f"theory-fingerprint-tests_v4{tag}.png"
    out_pdf = PLOT_DIR / f"theory-fingerprint-tests_v4{tag}.pdf"
    out_summary = PLOT_DIR / f"summary{tag}.json"

    temps = sorted([float(t.strip()) for t in args.temps.split(",") if t.strip()])

    if args.replot:
        if not out_pt.exists():
            print(f"Error: data file not found: {out_pt}")
            return
        payload = torch.load(out_pt, weights_only=False)
        data = payload["data"]
        temps = payload["config"]["temps"]
        print(f"Replotting from {out_pt}")
        make_plot(data, temps, out_png, out_pdf)
        print(f"saved: {out_png}")
        print(f"saved: {out_pdf}")
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device} seeds={seeds} temps={temps}")
    print("Running temperature fingerprint test...")
    data = run_experiment(seeds, temps, args.train_steps, args.train_lr,
                          args.batch_size, device)

    payload = {
        "config": {"seeds": seeds, "temps": temps, "train_steps": args.train_steps,
                   "train_lr": args.train_lr, "batch_size": args.batch_size},
        "data": data,
    }
    torch.save(payload, out_pt)
    make_plot(data, temps, out_png, out_pdf)
    write_summary(
        out_summary,
        build_summary(payload["data"], temps, payload["config"], out_pt, out_png, out_pdf, out_summary),
    )

    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
