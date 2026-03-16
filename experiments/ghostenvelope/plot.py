#!/usr/bin/env python3
"""Plot ghost envelope: 3x4 grid of Re/Im scatter with envelope bands.

Loads cache/ghostenvelope{tag}.pt and renders results/ghost-envelope{tag}.png.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ghostenv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cache"
PLOT_DIR = ROOT / "results"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "red": "#E3120B", "blue": "#006BA2", "teal": "#00A5A5",
    "gold": "#F4A100", "green": "#00843D", "dark": "#3D3D3D",
    "mid": "#767676", "light": "#D0D0D0",
}
COLORS = ["#E3120B", "#006BA2", "#00A5A5", "#F4A100"]
STAGES = ["Untrained", "Mid", "Final"]


def envBands(re, im, nbins=30):
    m = np.isfinite(re) & np.isfinite(im) & (im > 0)
    re, im = re[m], im[m]
    if re.size < 30:
        return np.array([]), np.array([]), np.array([])
    lo, hi = np.percentile(re, [5, 95])
    edges = np.linspace(lo, hi, nbins + 1)
    cen, low, high = [], [], []
    for i in range(nbins):
        b = (re >= edges[i]) & (re < edges[i + 1])
        if np.sum(b) < 8:
            continue
        cen.append(0.5 * (edges[i] + edges[i + 1]))
        low.append(np.percentile(im[b], 10))
        high.append(np.percentile(im[b], 90))
    return np.array(cen), np.array(low), np.array(high)


def plotOne(ax, snap, color, title):
    re = snap["re"]
    im = snap["im"]
    fin = np.isfinite(im)
    if not np.any(fin):
        ax.set_title(title, loc="left", fontsize=10)
        return
    cap = np.percentile(im[fin], 99)
    m = np.isfinite(re) & np.isfinite(im) & (im > 0) & (im < cap)
    rp, ip = re[m], im[m]
    if rp.size == 0:
        ax.set_title(title, loc="left", fontsize=10)
        return

    ax.scatter(rp, ip, s=8, alpha=0.2, color=color, edgecolors="none")
    cen, lo, hi = envBands(rp, ip)
    if cen.size:
        ax.plot(cen, lo, color=PALETTE["red"], lw=1.6, label="Q10")
        ax.plot(cen, hi, color=PALETTE["blue"], lw=1.4, ls="--", label="Q90")

    rho = snap["rhoA"]
    ax.axhline(rho, color=PALETTE["green"], lw=1.4, ls=":", label="rho_a")
    ax.axhline(0, color=PALETTE["dark"], lw=1.0)

    qx = np.percentile(rp, [2, 98])
    qy = np.percentile(ip, [2, 98])
    px = 0.15 * (qx[1] - qx[0] + 1e-6)
    py = 0.15 * (qy[1] - qy[0] + 1e-6)
    ax.set_xlim(qx[0] - px, qx[1] + px)
    ax.set_ylim(0, qy[1] + py)

    dist = np.sqrt(rp * rp + ip * ip)
    far = dist >= np.percentile(dist, 90)
    if np.any(far):
        ang = float(np.mean(np.degrees(np.arctan2(ip[far], np.abs(rp[far])))))
        ax.text(0.02, 0.95, f"far-angle={ang:.1f}\u00b0",
                transform=ax.transAxes, va="top", fontsize=7, color=PALETTE["mid"])

    acc = snap["acc"]
    ax.set_title(f"{title} ({100*acc:.0f}%)", loc="left", fontsize=10)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--tag", type=str, default="")
    args = pa.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    pt = DATA_DIR / f"ghostenvelope{tag}.pt"
    if not pt.exists():
        print(f"Not found: {pt}")
        return

    payload = torch.load(pt, weights_only=False)
    data = payload["data"]
    archs = payload["archs"]

    fig, axes = plt.subplots(3, len(archs), figsize=(5 * len(archs), 12))
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    })

    for col, arch in enumerate(archs):
        seeds = sorted(data[arch].keys())
        snaps = data[arch][seeds[0]]
        color = COLORS[col % len(COLORS)]
        for row, stage in enumerate(STAGES):
            ax = axes[row, col]
            ax.grid(axis="both", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plotOne(ax, snaps[row], color, f"{arch} {stage}")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            if row == 2:
                ax.set_xlabel("Re(ghost)")
            if col == 0:
                ax.set_ylabel("Im(ghost)")

    fig.suptitle("Ghost Envelope Structure Across Training",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outpng = PLOT_DIR / f"ghost-envelope{tag}.png"
    outpdf = PLOT_DIR / f"ghost-envelope{tag}.pdf"
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {outpng}")
    print(f"saved: {outpdf}")


if __name__ == "__main__":
    main()
