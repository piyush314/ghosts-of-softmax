#!/usr/bin/env python3
"""Publication-quality sigmoid plot showing x = π threshold."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ghosts.plotting import PALETTE, apply_plot_style, finish_figure

apply_plot_style(font_size=12, title_size=14, label_size=12, tick_size=11, legend_size=11)

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
ax.hlines(p_pi, x.min(), x_pi, color=PALETTE['mid_gray'], ls=':', lw=1.2, zorder=1)

# Mark the point
ax.scatter([x_pi], [p_pi], color=PALETTE['red'], s=60, zorder=4, edgecolor='white', linewidth=1.5)

# Annotations
ax.annotate(
    f'x = π\nσ(π) = {p_pi:.2f}',
    xy=(x_pi, p_pi),
    xytext=(x_pi * 2.5, p_pi - 0.12),
    fontsize=13,
    color=PALETTE['dark_gray'],
    ha='left',
    arrowprops=dict(arrowstyle='->', color=PALETTE['mid_gray'], lw=1.2),
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
ax.yaxis.grid(True, alpha=0.4, color=PALETTE['light_gray'])
ax.set_axisbelow(True)

# Title and subtitle
fig.suptitle(r'Complex Singularity at $i\pi$ Bounds Taylor Series to $|x| < \pi$',
             fontsize=13, fontweight='bold', x=0.02, ha='left')
ax.set_title('At x = π, prediction already 96% confident',
             loc='left', pad=4, fontsize=11, fontweight='normal',
             color=PALETTE['mid_gray'])

finish_figure(fig, rect=[0, 0, 1, 0.93])

# === SAVE ===
outdir = Path(__file__).resolve().parent / "results"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "sigmoid-pi.pdf", bbox_inches='tight', dpi=300)
fig.savefig(outdir / "sigmoid-pi.png", bbox_inches='tight', dpi=150)
print(f"Saved to {outdir / 'sigmoid-pi.pdf'}")
plt.show()
