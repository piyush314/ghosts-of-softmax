#!/usr/bin/env python3
"""Publication-quality sigmoid plot showing x = π threshold."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === PALETTE (Economist style) ===
PALETTE = {
    'blue': '#006BA2',
    'red': '#E3120B',
    'green': '#00843D',
    'gold': '#D4A017',
    'dark': '#3D3D3D',
    'mid': '#767676',
    'light': '#D0D0D0',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['dark'],
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# === DATA ===
x = np.logspace(-2, 2, 500)
sigmoid = 1 / (1 + np.exp(-x))  # p(x) = 1/(1+e^{-x}) = e^x/(1+e^x)

# Key point: x = π
x_pi = np.pi
p_pi = 1 / (1 + np.exp(-x_pi))

# === FIGURE ===
fig, ax = plt.subplots(figsize=(7, 4.5))

# Main curve
ax.plot(x, sigmoid, color=PALETTE['blue'], lw=2.5, zorder=3)

# Vertical line at x = π
ax.axvline(x_pi, color=PALETTE['red'], ls='--', lw=1.8, zorder=2, alpha=0.8)

# Horizontal line showing p(π)
ax.hlines(p_pi, x.min(), x_pi, color=PALETTE['mid'], ls=':', lw=1.2, zorder=1)

# Mark the point
ax.scatter([x_pi], [p_pi], color=PALETTE['red'], s=60, zorder=4, edgecolor='white', linewidth=1.5)

# Annotations
ax.annotate(
    f'x = π\nσ(π) = {p_pi:.2f}',
    xy=(x_pi, p_pi),
    xytext=(x_pi * 2.5, p_pi - 0.12),
    fontsize=13,
    color=PALETTE['dark'],
    ha='left',
    arrowprops=dict(arrowstyle='->', color=PALETTE['mid'], lw=1.2),
)

# Shaded regions
ax.axvspan(x.min(), x_pi, alpha=0.06, color=PALETTE['green'], zorder=0)
ax.axvspan(x_pi, x.max(), alpha=0.04, color=PALETTE['red'], zorder=0)

# Region labels
ax.text(0.15, 0.55, 'Taylor\nconverges', fontsize=12, color=PALETTE['green'],
        ha='center', fontweight='bold', alpha=0.9)
ax.text(20, 0.55, 'diverges', fontsize=12, color=PALETTE['red'],
        ha='center', fontweight='bold', alpha=0.9)

# Axes
ax.set_xscale('log')
ax.set_xlabel('Margin x (log scale)', fontsize=13)
ax.set_ylabel(r'$\sigma(x) = 1/(1+e^{-x})$', fontsize=13)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0.48, 1.02)

# Grid
ax.yaxis.grid(True, alpha=0.4, color=PALETTE['light'])
ax.set_axisbelow(True)

# Title and subtitle
fig.suptitle(r'Complex Singularity at $i\pi$ Bounds Taylor Series to $|x| < \pi$',
             fontsize=13, fontweight='bold', x=0.02, ha='left')
ax.set_title('At x = π, prediction already 96% confident',
             loc='left', pad=4, fontsize=11, fontweight='normal',
             color=PALETTE['mid'])

plt.tight_layout()

# === SAVE ===
outdir = Path(__file__).parent.parent / "paper/figures/plots"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "sigmoid-pi.pdf", bbox_inches='tight', dpi=300)
fig.savefig(outdir / "sigmoid-pi.png", bbox_inches='tight', dpi=150)
print(f"Saved to {outdir / 'sigmoid-pi.pdf'}")
plt.show()
