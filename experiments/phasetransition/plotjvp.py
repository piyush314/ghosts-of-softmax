#!/usr/bin/env python3
"""Publication-quality phase transition plot using exact JVP radius."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# === PALETTE (Economist) ===
PALETTE = {
    'red': '#E3120B',
    'blue': '#006BA2',
    'teal': '#00A5A5',
    'gold': '#F4A100',
    'purple': '#6F2DA8',
    'green': '#00843D',
    'dark_gray': '#3D3D3D',
    'mid_gray': '#767676',
    'light_gray': '#D0D0D0',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['dark_gray'],
    'axes.linewidth': 0.8,
    'grid.color': PALETTE['light_gray'],
    'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# === LOAD DATA ===
datapath = Path(__file__).parent.parent / "figures/ghosts_softmax"
data = np.load(datapath / "exp7e_rho_deltamax_a_jvp_fullacc_results.npz",
               allow_pickle=True)
r_values = data['r_values']

# Architecture config
archs = [
    ('Linear', PALETTE['red'], 2.0),      # protagonist - transitions at r=1
    ('MLP', PALETTE['gold'], 1.5),
    ('CNN', '#DC7633', 1.5),
    ('MLP+LayerNorm', PALETTE['blue'], 1.5),
    ('CNN+BatchNorm', PALETTE['green'], 1.5),
    ('TinyTransformer', PALETTE['purple'], 1.5),
]

# === BUILD TWO-PANEL FIGURE ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: Accuracy
ax1.axvspan(r_values.min(), 1.0, color=PALETTE['green'], alpha=0.06, zorder=0)
ax1.axvspan(1.0, r_values.max(), color=PALETTE['red'], alpha=0.04, zorder=0)

for arch, color, lw in archs:
    acc = data[f'{arch}_acc'] * 100
    std = data[f'{arch}_acc_std'] * 100
    ax1.plot(r_values, acc, color=color, lw=lw, label=arch)
    ax1.fill_between(r_values, acc - std, acc + std, color=color, alpha=0.12)

ax1.axvline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.2, zorder=5)
ax1.set_xscale('log')
ax1.set_xlabel(r'Normalized step $r = \tau / \rho_a$')
ax1.set_ylabel('Test accuracy retained (%)')
ax1.set_xlim(r_values.min(), r_values.max())
ax1.set_ylim(0, 105)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.yaxis.grid(True, alpha=0.7)
ax1.set_title(r"(A) Accuracy collapse near $r = 1$", loc='left', pad=8)
ax1.text(0.15, 92, 'safe', fontsize=9, color=PALETTE['green'], fontweight='bold')
ax1.text(3, 92, 'unsafe', fontsize=9, color=PALETTE['red'], fontweight='bold')

# Panel B: Loss inflation
for arch, color, lw in archs:
    loss = data[f'{arch}_loss']
    std = data[f'{arch}_loss_std']
    ax2.plot(r_values, loss, color=color, lw=lw, label=arch)
    ax2.fill_between(r_values, loss - std, loss + std, color=color, alpha=0.12)

ax2.axvline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.2, zorder=5)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Normalized step $r = \tau / \rho_a$')
ax2.set_ylabel('Loss inflation (× baseline)')
ax2.set_xlim(r_values.min(), r_values.max())
ax2.yaxis.grid(True, alpha=0.7)
ax2.set_title("(B) Loss explodes beyond r = 1", loc='left', pad=8)

# Direct labels for panel B (right edge)
label_y = {
    'Linear': 300, 'MLP': 8000, 'CNN': 3000,
    'MLP+LayerNorm': 25000, 'CNN+BatchNorm': 800000, 'TinyTransformer': 80000,
}
for arch, color, lw in archs:
    label = arch.replace('+LayerNorm', '+LN').replace('+BatchNorm', '+BN')
    ax2.text(r_values[-1] * 1.15, label_y[arch], label, fontsize=7,
             color=color, va='center', ha='left')

plt.subplots_adjust(right=0.85, wspace=0.25)

# Legend for panel A only
ax1.legend(loc='lower left', fontsize=7, ncol=2, framealpha=0.9)

# Suptitle
fig.suptitle(r"Phase Transition via JVP-Based $\rho_a$: Linear Model at $r = 1$",
             fontsize=12, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# === SAVE ===
outdir = Path(__file__).parent.parent / "paper/figures/plots"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "phase-jvp.pdf", bbox_inches='tight', dpi=300)
fig.savefig(outdir / "phase-jvp.png", bbox_inches='tight', dpi=150)
print(f"Saved to {outdir / 'phase-jvp.pdf'}")
