#!/usr/bin/env python3
"""Multi-seed transformer radii comparison (Figure 9).

Compares adaptive vs fixed learning rates on tiny transformer:
  - all-radii: min(rho_out, rho_attn, rho_ffn)
  - attn-out: min(rho_out, rho_attn)
  - output-only: rho_out only
  - fixed-1x: base LR
  - fixed-16x: 16× base LR

Outputs:
  - falsify/data/transformer_radii_multiseed.pt
  - paper/figures/plots/transformer-radii-compare.{png,pdf}
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import site
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-exp27")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    'blue': '#1565C0',
    'orange': '#D84315',
    'green': '#2E7D32',
    'purple': '#7B1FA2',
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
        raise FileNotFoundError("Could not find sklearn digits.csv.gz in site-packages")

    arr = np.loadtxt(gzip.open(csv_path, "rt"), delimiter=",", dtype=np.float32)
    X = arr[:, :-1] / 16.0
    y = arr[:, -1].astype(np.int64)
    return X, y


def train_test_split(X, y, test_size=0.3, seed=42):
    n = len(X)
    n_test = int(n * test_size)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# === MODEL ===

class TinySelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        w = torch.softmax(logits, dim=-1)
        y = torch.matmul(w, v)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y), logits if return_logits else None


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, return_attn_logits=False, return_ffn_preact=False):
        h = self.ln1(x)
        attn_out, attn_logits = self.attn(h, return_logits=return_attn_logits)
        x = x + attn_out
        h2 = self.ln2(x)
        ffn_preact = self.fc1(h2)
        x = x + self.fc2(F.gelu(ffn_preact))
        return x, attn_logits, ffn_preact if return_ffn_preact else None


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 32, depth: int = 2, n_head: int = 4):
        super().__init__()
        self.embed_row = nn.Linear(8, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 9, d_model))
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_head, 4 * d_model)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x, return_attn_logits=False, return_ffn_preact=False):
        b = x.shape[0]
        t = self.embed_row(x.view(b, 8, 8))
        h = torch.cat([self.cls.expand(b, -1, -1), t], dim=1) + self.pos
        attn_all, ffn_all = [], []
        for blk in self.blocks:
            h, attn_l, ffn_p = blk(h, return_attn_logits, return_ffn_preact)
            if attn_l is not None:
                attn_all.append(attn_l)
            if ffn_p is not None:
                ffn_all.append(ffn_p)
        return self.head(self.norm(h[:, 0])), attn_all, ffn_all


def adam_direction(model: nn.Module, opt: torch.optim.Optimizer) -> Tuple[Dict, float]:
    pdir_by_id = {}
    sq = 0.0
    for group in opt.param_groups:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()
            st = opt.state[p]
            exp_avg = st.get("exp_avg", torch.zeros_like(p))
            exp_avg_sq = st.get("exp_avg_sq", torch.zeros_like(p))
            step_obj = st.get("step", 0)
            step_val = int(step_obj.item()) if torch.is_tensor(step_obj) else int(step_obj)
            m = exp_avg * beta1 + g * (1.0 - beta1)
            v = exp_avg_sq * beta2 + (g * g) * (1.0 - beta2)
            bc1 = 1.0 - (beta1 ** (step_val + 1))
            bc2 = 1.0 - (beta2 ** (step_val + 1))
            pdir = (m / bc1) / (torch.sqrt(v / bc2) + eps)
            pdir_by_id[id(p)] = pdir
            sq += float((pdir ** 2).sum().item())
    pnorm = math.sqrt(sq) + 1e-12
    tangents = {}
    for n, p in model.named_parameters():
        if id(p) in pdir_by_id:
            tangents[n] = -pdir_by_id[id(p)] / pnorm
        else:
            tangents[n] = torch.zeros_like(p)
    return tangents, pnorm


def rho_components(model: TinyTransformer, xb: torch.Tensor, tangents: Dict) -> Tuple[float, float, float, float]:
    params = {n: p for n, p in model.named_parameters()}

    def f(pdict):
        logits, attn_logits, ffn_preact = functional_call(model, pdict, (xb, True, True))
        attn_stack = torch.stack(attn_logits, dim=0) if attn_logits else logits.new_zeros((0,))
        ffn_stack = torch.stack(ffn_preact, dim=0) if ffn_preact else logits.new_zeros((0,))
        return logits, attn_stack, ffn_stack

    (_, _, ffn_preact), (dlogits, datt, dffn) = jvp(f, (params,), (tangents,))

    # Output radius
    spread_out = dlogits.max(dim=-1).values - dlogits.min(dim=-1).values
    rho_out = math.pi / max(float(spread_out.max().item()), 1e-12)

    # Attention radius
    if datt.numel() > 0:
        dmax = datt.max(dim=-1).values
        dmin = datt.min(dim=-1).values
        rho_attn = math.pi / max(float((dmax - dmin).max().item()), 1e-12)
    else:
        rho_attn = float("inf")

    # FFN radius (kink-based)
    if dffn.numel() > 0:
        ratios = (ffn_preact.abs() / dffn.abs().clamp_min(1e-12)).flatten()
        rho_ffn = float(torch.quantile(ratios, 0.01).item())
    else:
        rho_ffn = float("inf")

    return min(rho_out, rho_attn, rho_ffn), rho_out, rho_attn, rho_ffn


def run_single(seed: int, mode: str, steps: int, lr_base: float, eval_every: int,
               device: torch.device) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=42)
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte = torch.tensor(yte, dtype=torch.long, device=device)

    model = TinyTransformer(d_model=32, depth=2, n_head=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-4)

    # Parse mode
    if mode == "all-radii":
        use_rho, rho_mode = True, "all"
    elif mode == "attn-out":
        use_rho, rho_mode = True, "attn_out"
    elif mode == "output-only":
        use_rho, rho_mode = True, "output"
    elif mode.startswith("fixed-"):
        use_rho = False
        multiplier = float(mode.split("-")[1].replace("x", ""))
        fixed_lr = lr_base * multiplier
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = {"loss": [], "acc": [], "lr_eff": [], "rho_out": [], "rho_attn": [],
               "rho_ffn": [], "rho_min": []}

    for step in range(steps + 1):
        model.train()
        logits, _, _ = model(xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if use_rho:
            tangents, pnorm = adam_direction(model, opt)
            rho_min, rho_out, rho_attn, rho_ffn = rho_components(model, xtr, tangents)
            if rho_mode == "all":
                rho_used = rho_min
            elif rho_mode == "attn_out":
                rho_used = min(rho_out, rho_attn)
            else:  # output
                rho_used = rho_out
            lr_eff = rho_used / pnorm
        else:
            lr_eff = fixed_lr
            rho_out = rho_attn = rho_ffn = rho_min = float("nan")

        for pg in opt.param_groups:
            pg["lr"] = lr_eff
        opt.step()

        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                acc = (model(xte)[0].argmax(1) == yte).float().mean().item()
            results["loss"].append(loss.item())
            results["acc"].append(acc)
            results["lr_eff"].append(lr_eff)
            results["rho_out"].append(rho_out)
            results["rho_attn"].append(rho_attn)
            results["rho_ffn"].append(rho_ffn)
            results["rho_min"].append(rho_min)

    return results


def run_experiment(seeds: List[int], modes: List[str], steps: int, lr_base: float,
                   eval_every: int, device: torch.device) -> Dict:
    data = {mode: {} for mode in modes}
    for mode in modes:
        print(f"  {mode}")
        for i, seed in enumerate(seeds):
            print(f"    seed {seed} ({i+1}/{len(seeds)})")
            data[mode][seed] = run_single(seed, mode, steps, lr_base, eval_every, device)
    return data


def make_plot(data: Dict, modes: List[str], eval_every: int, out_png: Path,
              out_pdf: Path) -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    colors = {
        'all-radii': PALETTE['blue'],
        'attn-out': PALETTE['mid'],
        'output-only': PALETTE['orange'],
        'fixed-1x': PALETTE['green'],
        'fixed-16x': PALETTE['purple'],
    }

    def stats(arrs):
        arr = np.array(arrs)
        return np.median(arr, axis=0), np.percentile(arr, 25, axis=0), \
               np.percentile(arr, 75, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.subplots_adjust(top=0.85, wspace=0.35)
    fig.suptitle("Tiny Transformer on Digits: Adaptive vs Fixed LR (Multi-seed)",
                 fontsize=13, fontweight='bold')

    seeds = list(data[modes[0]].keys())
    n_points = len(data[modes[0]][seeds[0]]["loss"])
    x = np.arange(n_points) * eval_every

    # Panel 0: Loss
    ax = axes[0]
    for mode in modes:
        color = colors.get(mode, '#333')
        losses = [data[mode][s]["loss"] for s in seeds]
        med, q25, q75 = stats(losses)
        ax.fill_between(x, np.maximum(q25, 1e-6), np.maximum(q75, 1e-6),
                        alpha=0.15, color=color)
        ax.semilogy(x, np.maximum(med, 1e-6), color=color, lw=2, label=mode)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Loss", fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    # Panel 1: Test Accuracy
    ax = axes[1]
    for mode in modes:
        color = colors.get(mode, '#333')
        accs = [data[mode][s]["acc"] for s in seeds]
        med, q25, q75 = stats(accs)
        ax.fill_between(x, q25, q75, alpha=0.15, color=color)
        ax.plot(x, med, color=color, lw=2, label=mode)
    ax.set_xlabel("Step")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy", fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Effective LR
    ax = axes[2]
    for mode in modes:
        color = colors.get(mode, '#333')
        lrs = [data[mode][s]["lr_eff"] for s in seeds]
        med, q25, q75 = stats(lrs)
        ax.fill_between(x, np.maximum(q25, 1e-12), np.maximum(q75, 1e-12),
                        alpha=0.15, color=color)
        ax.semilogy(x, np.maximum(med, 1e-12), color=color, lw=2, label=mode)
    ax.set_xlabel("Step")
    ax.set_ylabel("Effective LR")
    ax.set_title("Effective Learning Rate", fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed transformer radii test.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--modes", type=str, default="all-radii,output-only,fixed-1x,fixed-16x")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr-base", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_pt = DATA_DIR / f"transformer_radii_multiseed{tag}.pt"
    out_png = PLOT_DIR / f"transformer-radii-compare{tag}.png"
    out_pdf = PLOT_DIR / f"transformer-radii-compare{tag}.pdf"

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.replot:
        if not out_pt.exists():
            print(f"Error: data file not found: {out_pt}")
            return
        payload = torch.load(out_pt, weights_only=False)
        data = payload["data"]
        modes = payload["config"]["modes"]
        eval_every = payload["config"]["eval_every"]
        print(f"Replotting from {out_pt}")
        make_plot(data, modes, eval_every, out_png, out_pdf)
        print(f"saved: {out_png}")
        print(f"saved: {out_pdf}")
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device} seeds={seeds} modes={modes}")
    print(f"steps={args.steps} lr_base={args.lr_base} eval_every={args.eval_every}")
    print("Running transformer radii comparison...")
    data = run_experiment(seeds, modes, args.steps, args.lr_base, args.eval_every, device)

    payload = {
        "config": {
            "seeds": seeds, "modes": modes, "steps": args.steps,
            "lr_base": args.lr_base, "eval_every": args.eval_every,
        },
        "data": data,
    }
    torch.save(payload, out_pt)
    make_plot(data, modes, args.eval_every, out_png, out_pdf)

    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")

    # Print final accuracies
    print("\nFinal accuracies:")
    for mode in modes:
        accs = [data[mode][s]["acc"][-1] for s in seeds]
        print(f"  {mode}: {np.median(accs)*100:.1f}% (IQR: {np.percentile(accs,25)*100:.1f}-{np.percentile(accs,75)*100:.1f}%)")


if __name__ == "__main__":
    main()
