#!/usr/bin/env python3
"""Ghost envelope experiment: Re/Im scatter + envelope bands.

Trains 4 architectures on digits, snapshots ghost geometry
at 3 stages (init, mid, final), saves for plotting.

Outputs:
  - cache/ghostenvelope{tag}.pt
  - results/ghost-envelope{tag}.{png,pdf}
  - results/summary{tag}.json
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ghostenv")

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

torch.set_num_threads(1)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Allow importing models.py from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import ARCHS, TinyTfm

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cache"
PLOT_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = ROOT.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ghosts.reporting import repo_relpath, scalar_stats, write_summary
from plot import make_plot


def loadDigits() -> Tuple[np.ndarray, np.ndarray]:
    candidates: List[Path] = []
    for sp in site.getsitepackages():
        p = Path(sp) / "sklearn" / "datasets" / "data" / "digits.csv.gz"
        candidates.append(p)
    usp = site.getusersitepackages()
    if usp:
        p = Path(usp) / "sklearn" / "datasets" / "data" / "digits.csv.gz"
        candidates.append(p)
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise FileNotFoundError("Could not find sklearn digits.csv.gz")
    arr = np.loadtxt(gzip.open(path, "rt"), delimiter=",", dtype=np.float32)
    return arr[:, :-1] / 16.0, arr[:, -1].astype(np.int64)


def stratSplit(X, y, ratio=0.3, seed=42):
    rng = np.random.default_rng(seed)
    trn, tst = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = int(round(ratio * len(idx)))
        tst.append(idx[:n])
        trn.append(idx[n:])
    trn, tst = np.concatenate(trn), np.concatenate(tst)
    rng.shuffle(trn)
    rng.shuffle(tst)
    return X[trn], X[tst], y[trn], y[tst]


def ghostStats(logits, slopes, y):
    n, c = logits.shape
    da = np.max(slopes, axis=1) - np.min(slopes, axis=1)
    pred = np.argmax(logits, axis=1)
    re = np.full(n, np.nan)
    im = np.full(n, np.inf)
    margin = np.zeros(n)
    correct = pred == y
    for i in range(n):
        yi = int(y[i])
        oth = logits[i].copy()
        oth[yi] = -np.inf
        cc = int(np.argmax(oth))
        delta = float(logits[i, yi] - logits[i, cc])
        s = float(slopes[i, cc] - slopes[i, yi])
        margin[i] = delta
        if abs(s) < 1e-12:
            continue
        re[i] = delta / s
        im[i] = math.pi / abs(s)
    rhoA = math.pi / max(float(np.max(da)), 1e-12)
    return {"re": re, "im": im, "margin": margin,
            "correct": correct, "rhoA": rhoA,
            "acc": float(np.mean(correct))}


def logitsSlopesJvp(model, Xeval, yeval):
    isTfm = isinstance(model, TinyTfm)
    was = model.training
    model.train() if isTfm else model.eval()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(Xeval), yeval)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    gnorm = float(torch.sqrt(sum((g*g).sum() for g in grads.values())).item())
    tans = {n: -g / max(gnorm, 1e-12) for n, g in grads.items()}
    params = {n: p.detach() for n, p in model.named_parameters()}

    def f(pd):
        return functional_call(model, pd, (Xeval,))

    logits = f(params).detach()
    _, slopes = jvp(f, (params,), (tans,))
    model.train() if was else model.eval()
    return logits.cpu().numpy(), slopes.detach().cpu().numpy()


def runArch(name, mkModel, lr, steps, Xtr, ytr, Xeval, yeval, seed):
    torch.manual_seed(seed)
    model = mkModel()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    snaps = [0, steps // 4, steps]
    results = []
    for step in range(steps + 1):
        if step in snaps:
            lg, sl = logitsSlopesJvp(model, Xeval, yeval)
            gs = ghostStats(lg, sl, yeval.cpu().numpy())
            gs["step"] = step
            results.append(gs)
        if step == steps:
            break
        idx = rng.choice(n, size=128, replace=False)
        model.train()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    return results


def build_summary(data: Dict, archs: List[str], config: Dict,
                  out_pt: Path, out_png: Path, out_pdf: Path,
                  out_summary: Path) -> Dict:
    summary = {
        "experiment_id": "ghostenvelope",
        "config": config,
        "artifacts": {
            "data": repo_relpath(out_pt, REPO_ROOT),
            "plot_png": repo_relpath(out_png, REPO_ROOT),
            "plot_pdf": repo_relpath(out_pdf, REPO_ROOT),
            "summary_json": repo_relpath(out_summary, REPO_ROOT),
        },
        "architectures": {},
    }
    for arch in archs:
        seeds = sorted(data[arch].keys())
        stage_names = ["init", "mid", "final"]
        stages = {}
        for idx, stage_name in enumerate(stage_names):
            snaps = [data[arch][seed][idx] for seed in seeds]
            rho_vals = [float(snap["rhoA"]) for snap in snaps]
            acc_vals = [float(snap["acc"]) for snap in snaps]
            finite_im = []
            for snap in snaps:
                im = np.asarray(snap["im"], dtype=float)
                finite_im.extend(im[np.isfinite(im) & (im > 0)].tolist())
            stages[stage_name] = {
                "rho_a": scalar_stats(rho_vals),
                "accuracy": scalar_stats(acc_vals),
                "ghost_imag": scalar_stats(finite_im),
            }
        summary["architectures"][arch] = stages
    return summary


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--seeds", type=str, default="0")
    pa.add_argument("--tag", type=str, default="")
    args = pa.parse_args()

    if args.smoke:
        args.tag = args.tag or "smoke"
    tag = f"_{args.tag}" if args.tag else ""
    seeds = [int(s) for s in args.seeds.split(",")]
    out_pt = DATA_DIR / f"ghostenvelope{tag}.pt"
    out_png = PLOT_DIR / f"ghost-envelope{tag}.png"
    out_pdf = PLOT_DIR / f"ghost-envelope{tag}.pdf"
    out_summary = PLOT_DIR / f"summary{tag}.json"

    X, y = loadDigits()
    Xtr, Xte, ytr, yte = stratSplit(X, y)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)

    rng = np.random.default_rng(42)
    neval = min(768, len(Xtr))
    ei = rng.choice(len(Xtr), size=neval, replace=False)
    Xeval, yeval = Xtr[ei], ytr[ei]

    data = {}
    for aname, (cls, lr, steps) in ARCHS.items():
        if args.smoke:
            steps = 20
        print(f"[{aname}] steps={steps} seeds={seeds}")
        data[aname] = {}
        for seed in seeds:
            data[aname][seed] = runArch(
                aname, cls, lr, steps, Xtr, ytr, Xeval, yeval, seed,
            )
            acc = data[aname][seed][-1]["acc"]
            print(f"  seed={seed} final_acc={acc:.3f}")

    payload = {
        "config": {"seeds": seeds, "smoke": bool(args.smoke)},
        "data": data,
        "archs": list(ARCHS.keys()),
    }
    torch.save(payload, out_pt)
    make_plot(payload, out_png, out_pdf)
    write_summary(
        out_summary,
        build_summary(payload["data"], payload["archs"], payload["config"],
                      out_pt, out_png, out_pdf, out_summary),
    )
    print(f"saved: {out_pt}")
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")
    print(f"saved: {out_summary}")


if __name__ == "__main__":
    main()
