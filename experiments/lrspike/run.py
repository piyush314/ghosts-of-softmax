#!/usr/bin/env python3
"""Multi-seed LR spike test: Plain Adam vs rho-controller.

Replicates Figure 4 (lrsweep-jvp) with multiple seeds for error bars.

Protocol:
  - MLP on Digits dataset
  - 3 base LRs: 1e-4, 1e-3, 1e-2
  - 1000x spike at step 50
  - 200 steps total
  - Compare: plain Adam vs rho_a-controller

Outputs:
  - falsify/data/lrsweep_multiseed.pt
  - paper/figures/plots/lrsweep-jvp-multiseed.{png,pdf}
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import site
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-exp24")

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
DATA_DIR = ROOT / "falsify" / "data"
PLOT_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'red': '#E3120B',
    'gold': '#F4A100',
    'green': '#00843D',
    'blue': '#1E5AA8',
    'dark': '#3D3D3D',
    'mid': '#767676',
    'light': '#D0D0D0',
}


def load_digits_from_csv() -> Tuple[np.ndarray, np.ndarray]:
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


def stratified_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0):
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


class SimpleMLP(nn.Module):
    """2-layer MLP matching original Figure 4 setup."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        return self.fc2(F.relu(self.fc1(x)))


class AdamCore:
    def __init__(self, params: Iterable[torch.Tensor], betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.state: Dict[torch.Tensor, Dict[str, torch.Tensor]] = {}

    def _state(self, p: torch.Tensor):
        if p not in self.state:
            self.state[p] = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
        return self.state[p]

    @torch.no_grad()
    def compute_p(self) -> Tuple[List[torch.Tensor], float]:
        self.t += 1
        b1t, b2t = self.b1 ** self.t, self.b2 ** self.t
        plist, sqsum = [], 0.0
        for p in self.params:
            g = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
            st = self._state(p)
            st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mhat = st["m"] / (1 - b1t)
            vhat = st["v"] / (1 - b2t)
            pd = mhat / (vhat.sqrt() + self.eps)
            plist.append(pd)
            sqsum += float(pd.pow(2).sum().item())
        return plist, math.sqrt(sqsum) + 1e-12

    @torch.no_grad()
    def apply(self, plist: List[torch.Tensor], lr: float):
        for p, pd in zip(self.params, plist):
            p.add_(pd, alpha=-lr)


def compute_rho_jvp(model: nn.Module, X: torch.Tensor, plist: List[torch.Tensor], pnorm: float) -> float:
    """Compute rho_a = pi / max_spread via JVP."""
    pu = [pp / pnorm for pp in plist]
    pdict = dict(model.named_parameters())
    tdict = {n: u for (n, _), u in zip(model.named_parameters(), pu)}
    was_training = model.training
    model.eval()

    def fwd(pd):
        return functional_call(model, pd, (X,))

    _, dlogits = jvp(fwd, (pdict,), (tdict,))
    if was_training:
        model.train()

    spread = dlogits.max(1).values - dlogits.min(1).values
    da_max = float(spread.max().item())
    return math.pi / (da_max + 1e-12)


@torch.no_grad()
def test_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    acc = float((model(X).argmax(1) == y).float().mean().item())
    if was_training:
        model.train()
    return acc


def batch_index_iter(n: int, bs: int, seed: int):
    rng = np.random.default_rng(seed)
    while True:
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            chunk = idx[i : i + bs]
            if len(chunk) > 0:
                yield torch.from_numpy(chunk).long()


def run_one(
    seed: int,
    base_lr: float,
    mode: str,  # "plain" or "rho_ctrl"
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    steps: int = 200,
    batch_size: int = 64,
    spike_at: int = 50,
    spike_mul: float = 1000.0,
) -> Dict[str, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleMLP().to(Xtr.device)
    opt = AdamCore(model.parameters())
    bit = batch_index_iter(len(Xtr), batch_size, seed + 11)

    log = {"loss": [], "acc": [], "tau": [], "rhoA": []}

    for t in range(steps):
        ib = next(bit)
        xb, yb = Xtr[ib], ytr[ib]
        lr_now = base_lr * (spike_mul if t >= spike_at else 1.0)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            pad = steps - t
            for k in log:
                log[k] += [float("nan")] * pad
            break

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        plist, pnorm = opt.compute_p()

        rhoA = compute_rho_jvp(model, Xtr, plist, pnorm)
        tau_raw = lr_now * pnorm

        if mode == "rho_ctrl" and tau_raw > rhoA:
            lr_eff = rhoA / pnorm
        else:
            lr_eff = lr_now

        tau_eff = lr_eff * pnorm

        log["loss"].append(float(loss.item()))
        log["acc"].append(test_acc(model, Xte, yte))
        log["tau"].append(float(tau_eff))
        log["rhoA"].append(float(rhoA))

        opt.apply(plist, lr_eff)

    return log


def run_experiment(
    seeds: List[int],
    lrs: List[float],
    device: torch.device,
    steps: int = 200,
    spike_at: int = 50,
    spike_mul: float = 1000.0,
    print_every: int = 50,
) -> Dict:
    X, y = load_digits_from_csv()
    all_data = {}

    for lr in lrs:
        lr_key = f"{lr:.0e}"
        all_data[lr_key] = {"plain": {}, "rho_ctrl": {}}

        for mode in ["plain", "rho_ctrl"]:
            all_data[lr_key][mode] = {
                "loss": [], "acc": [], "tau": [], "rhoA": []
            }

            for seed in seeds:
                Xtr_np, Xte_np, ytr_np, yte_np = stratified_split(X, y, 0.2, seed)
                Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=device)
                Xte = torch.tensor(Xte_np, dtype=torch.float32, device=device)
                ytr = torch.tensor(ytr_np, dtype=torch.long, device=device)
                yte = torch.tensor(yte_np, dtype=torch.long, device=device)

                log = run_one(seed, lr, mode, Xtr, ytr, Xte, yte,
                              steps=steps, spike_at=spike_at, spike_mul=spike_mul)

                for k in log:
                    all_data[lr_key][mode][k].append(log[k])

                if print_every > 0:
                    final_loss = log["loss"][-1] if log["loss"] else float("nan")
                    final_acc = log["acc"][-1] if log["acc"] else float("nan")
                    print(f"lr={lr_key} mode={mode:9s} seed={seed} "
                          f"final_loss={final_loss:.4f} final_acc={final_acc:.3f}")

    return all_data


def make_plot(
    data: Dict,
    lrs: List[str],
    steps: int,
    spike_at: int,
    out_png: Path,
    out_pdf: Path,
):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': PALETTE['dark'],
        'axes.linewidth': 0.8,
    })

    MESSAGES = {
        '1e-04': 'Adam recovers — spike stays within safe radius',
        '1e-03': 'Accuracy drops — exceeds radius, partial recovery',
        '1e-02': 'Learning erased — strong spike far exceeds radius',
    }

    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.22, 1, 0.22, 1, 0.22, 1],
                           hspace=0.05, wspace=0.28)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.09, right=0.97)

    fig.suptitle('Learning Rate Spike Test: Plain Adam vs ρ-Controller (multi-seed)',
                 fontsize=15, fontweight='bold', y=0.97)

    fig.text(0.30, 0.925, '— Plain Adam', fontsize=11, color=PALETTE['red'],
             fontweight='bold', ha='left')
    fig.text(0.50, 0.925, '— ρₐ controller', fontsize=11, color=PALETTE['gold'],
             fontweight='bold', ha='left')
    fig.text(0.72, 0.925, '█ ρₐ = π/Δₐ (safe zone)', fontsize=11,
             color=PALETTE['green'], ha='left')

    fig.text(0.22, 0.895, 'Training Loss', fontsize=12, fontweight='bold',
             ha='center', color=PALETTE['dark'])
    fig.text(0.53, 0.895, 'Test Accuracy', fontsize=12, fontweight='bold',
             ha='center', color=PALETTE['dark'])
    fig.text(0.84, 0.895, 'Step τ vs Radius ρₐ', fontsize=12, fontweight='bold',
             ha='center', color=PALETTE['dark'])

    x = np.arange(steps)

    for i, lr in enumerate(lrs):
        textrow = i * 2
        plotrow = i * 2 + 1

        # Text row
        ax_text = fig.add_subplot(gs[textrow, :])
        ax_text.set_axis_off()
        exp = int(np.log10(float(lr)))
        msg = MESSAGES.get(lr, '')
        combined = rf'$\eta_0 = 10^{{{exp}}}$:  {msg}'
        ax_text.text(0.5, 0.4, combined, transform=ax_text.transAxes,
                     fontsize=14, fontweight='bold', color=PALETTE['blue'],
                     ha='center', va='center')

        d = data[lr]

        # Helper to get median and IQR
        def stats(arr_list):
            arr = np.array(arr_list, dtype=float)
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanpercentile(arr, 25, axis=0)
            q75 = np.nanpercentile(arr, 75, axis=0)
            return med, q25, q75

        # Column 0: Loss
        ax = fig.add_subplot(gs[plotrow, 0])
        for mode, color in [("plain", PALETTE['red']), ("rho_ctrl", PALETTE['gold'])]:
            med, q25, q75 = stats(d[mode]["loss"])
            ax.semilogy(x, np.maximum(med, 1e-9), color=color, lw=2)
            ax.fill_between(x, np.maximum(q25, 1e-9), np.maximum(q75, 1e-9),
                            color=color, alpha=0.15)
        ax.axvline(spike_at, color=PALETTE['mid'], ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0, steps)
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE['light'])
        if i == 0:
            ax.annotate(r'$1000\times$ spike', xy=(spike_at, 0.6),
                        xycoords=('data', 'axes fraction'),
                        xytext=(15, 5), textcoords='offset points',
                        fontsize=9, color=PALETTE['dark'],
                        arrowprops=dict(arrowstyle='->', color=PALETTE['mid']))
        if i == 2:
            ax.set_xlabel('Step', fontsize=11)
        else:
            ax.set_xticklabels([])

        # Column 1: Accuracy
        ax = fig.add_subplot(gs[plotrow, 1])
        for mode, color in [("plain", PALETTE['red']), ("rho_ctrl", PALETTE['gold'])]:
            med, q25, q75 = stats(d[mode]["acc"])
            ax.plot(x, med, color=color, lw=2)
            ax.fill_between(x, np.clip(q25, 0, 1), np.clip(q75, 0, 1),
                            color=color, alpha=0.15)
        ax.axvline(spike_at, color=PALETTE['mid'], ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0, steps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0%', '50%', '100%'])
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE['light'])
        if i == 2:
            ax.set_xlabel('Step', fontsize=11)
        else:
            ax.set_xticklabels([])

        # Column 2: Step vs Radius
        ax = fig.add_subplot(gs[plotrow, 2])
        rho_med, rho_q25, rho_q75 = stats(d["rho_ctrl"]["rhoA"])
        tau_plain_med, tau_plain_q25, tau_plain_q75 = stats(d["plain"]["tau"])
        tau_ctrl_med, tau_ctrl_q25, tau_ctrl_q75 = stats(d["rho_ctrl"]["tau"])

        # rho IQR band + median
        ax.fill_between(x, np.maximum(rho_q25, 1e-12), np.maximum(rho_q75, 1e-12),
                        alpha=0.2, color=PALETTE['green'])
        ax.semilogy(x, np.maximum(rho_med, 1e-12), color=PALETTE['green'], lw=2,
                    label=r'$\rho_a$')
        # tau plain IQR band + median
        ax.fill_between(x, np.maximum(tau_plain_q25, 1e-12),
                        np.maximum(tau_plain_q75, 1e-12),
                        alpha=0.15, color=PALETTE['red'])
        ax.semilogy(x, np.maximum(tau_plain_med, 1e-12), color=PALETTE['red'], lw=2,
                    label=r'$\tau$ plain')
        # tau ctrl IQR band + median
        ax.fill_between(x, np.maximum(tau_ctrl_q25, 1e-12),
                        np.maximum(tau_ctrl_q75, 1e-12),
                        alpha=0.15, color=PALETTE['gold'])
        ax.semilogy(x, np.maximum(tau_ctrl_med, 1e-12), color=PALETTE['gold'], lw=2,
                    label=r'$\tau$ ctrl')
        ax.axvline(spike_at, color=PALETTE['mid'], ls='--', lw=1, alpha=0.7)
        ax.set_xlim(0, steps)

        all_vals = np.concatenate([
            rho_q25, rho_med, rho_q75,
            tau_plain_q25, tau_plain_med, tau_plain_q75,
            tau_ctrl_q25, tau_ctrl_med, tau_ctrl_q75
        ])
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        if len(all_vals) > 0:
            ymin, ymax = all_vals.min() / 2, all_vals.max() * 2
            ax.set_ylim(ymin, ymax)
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE['light'])

        if i == 2:
            ax.set_xlabel('Step', fontsize=11)
        else:
            ax.set_xticklabels([])

    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed LR spike test.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--lrs", type=str, default="1e-4,1e-3,1e-2")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--spike-at", type=int, default=50)
    parser.add_argument("--spike-mul", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--replot", action="store_true",
                        help="Replot from existing data without re-running experiment")
    args = parser.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_pt = DATA_DIR / f"lrsweep_multiseed{tag}.pt"
    out_png = PLOT_DIR / f"lrsweep-jvp-multiseed{tag}.png"
    out_pdf = PLOT_DIR / f"lrsweep-jvp-multiseed{tag}.pdf"

    if args.replot:
        if not out_pt.exists():
            print(f"Error: data file not found: {out_pt}")
            return
        payload = torch.load(out_pt, weights_only=False)
        data = payload["data"]
        lr_keys = payload["config"]["lrs"]
        steps = payload["config"]["steps"]
        spike_at = payload["config"]["spike_at"]
        print(f"Replotting from {out_pt}")
        make_plot(data, lr_keys, steps, spike_at, out_png, out_pdf)
        print(f"saved plot: {out_png}")
        print(f"saved plot: {out_pdf}")
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lrs = [float(s.strip()) for s in args.lrs.split(",") if s.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device} seeds={seeds} lrs={lrs} steps={args.steps} "
          f"spike_at={args.spike_at} spike_mul={args.spike_mul}")

    data = run_experiment(seeds, lrs, device, args.steps,
                          args.spike_at, args.spike_mul, args.print_every)

    payload = {
        "config": {
            "seeds": seeds,
            "lrs": [f"{lr:.0e}" for lr in lrs],
            "steps": args.steps,
            "spike_at": args.spike_at,
            "spike_mul": args.spike_mul,
        },
        "data": data,
    }
    torch.save(payload, out_pt)

    lr_keys = [f"{lr:.0e}" for lr in lrs]
    make_plot(data, lr_keys, args.steps, args.spike_at, out_png, out_pdf)

    print(f"saved data: {out_pt}")
    print(f"saved plot: {out_png}")
    print(f"saved plot: {out_pdf}")


if __name__ == "__main__":
    main()
