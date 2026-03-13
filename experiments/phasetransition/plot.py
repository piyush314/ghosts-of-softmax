#!/usr/bin/env python3
"""Publication-quality phase transition plot following Economist style."""
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

# === STYLE SETUP ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 15,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['dark_gray'],
    'axes.linewidth': 0.8,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.color': PALETTE['dark_gray'],
    'ytick.color': PALETTE['dark_gray'],
    'grid.color': PALETTE['light_gray'],
    'grid.linewidth': 0.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'legend.frameon': False,
    'legend.fontsize': 11,
})

# === LOAD DATA ===
datapath = Path(__file__).resolve().parent / "results"
data = np.load(datapath / "exp7b_rho_deltamax_results.npz", allow_pickle=True)
r_values = data['r_values']

# Architecture config: name, color, linewidth (protagonist = thick)
archs = [
    ('Linear', PALETTE['red'], 1.8),
    ('MLP', PALETTE['gold'], 1.5),
    ('CNN', '#DC7633', 1.5),  # orange
    ('MLP+LayerNorm', PALETTE['blue'], 1.8),
    ('CNN+BatchNorm', PALETTE['green'], 1.8),
    ('TinyTransformer', PALETTE['purple'], 1.8),
]

# === BUILD FIGURE ===
fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Safe zone shading (r < 1)
ax.axvspan(r_values.min(), 1.0, color=PALETTE['green'], alpha=0.06, zorder=0)

# Plot each architecture
for arch, color, lw in archs:
    acc = data[f'{arch}_acc'] * 100
    std = data[f'{arch}_acc_std'] * 100
    ax.plot(r_values, acc, color=color, lw=lw, label=arch)
    ax.fill_between(r_values, acc - std, acc + std, color=color, alpha=0.12)

# r=1 boundary
ax.axvline(1.0, color=PALETTE['dark_gray'], ls='--', lw=1.2, zorder=5)

# Direct labels at right edge with manual y-offsets to avoid overlap
plt.subplots_adjust(right=0.68)
label_positions = {
    'Linear': 22,
    'MLP': 14,
    'CNN': 4,
    'MLP+LayerNorm': 42,
    'TinyTransformer': 32,
    'CNN+BatchNorm': 52,
}
for arch, color, lw in archs:
    yval = label_positions[arch]
    label = arch.replace('+LayerNorm', '+LN').replace('+BatchNorm', '+BN')
    ax.text(r_values[-1] * 1.2, yval, label, fontsize=11, color=color,
            va='center', ha='left', fontweight='bold' if lw > 1.6 else 'normal',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.8))

# Annotations
ax.text(0.14, 94, 'Safe zone', fontsize=13, color=PALETTE['green'],
        fontweight='bold', ha='center')
ax.text(0.14, 88, r'($r < 1$)', fontsize=12, color=PALETTE['green'], ha='center')
ax.text(1.15, 8, r'$r = 1$', fontsize=13, color=PALETTE['dark_gray'],
        rotation=90, va='bottom')

# Title (declarative headline)
ax.set_title("A Single Step Can Wipe Learned Accuracy", loc='left', pad=18)
ax.text(0, 1.005, "Test accuracy retained after one gradient step, by architecture",
        transform=ax.transAxes, fontsize=12, color=PALETTE['mid_gray'], va='bottom')

# Axes
ax.set_xscale('log')
ax.set_xlabel(r'Normalized step $r = \tau / \rho_a$')
ax.set_ylabel('Test accuracy (%)')
ax.set_xlim(r_values.min(), r_values.max())
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))

# Grid (horizontal only for line chart)
ax.yaxis.grid(True, linestyle='-', alpha=0.7)
ax.xaxis.grid(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])

# === SAVE ===
outdir = Path(__file__).resolve().parent / "results"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "phase-transition.pdf", bbox_inches='tight', dpi=300)
fig.savefig(outdir / "phase-transition.png", bbox_inches='tight', dpi=150)
print(f"Saved to {outdir / 'phase-transition.pdf'}")
