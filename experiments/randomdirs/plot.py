#!/usr/bin/env python3
"""Publication-quality plot: Random direction sweeps."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === PALETTE (Economist style) ===
PALETTE = {
    'red': '#E3120B',
    'blue': '#006BA2',
    'green': '#00843D',
    'dark': '#3D3D3D',
    'mid': '#767676',
    'light': '#D0D0D0',
    'gold': '#C4A000',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': PALETTE['dark'],
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# === LOAD DATA ===
datapath = Path(__file__).resolve().parent / "results"
data = np.load(datapath / "exp11_random_direction_sweep_fixed_results.npz",
               allow_pickle=True)

stages = ['early', 'mid', 'late']
colTitles = ['Early (73% acc)', 'Mid (91% acc)', 'Late (94% acc)']
rowLabels = ['Loss Ratio', 'Flip Fraction']

# === FIGURE: 2 rows x 3 cols ===
fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True)

for col, (stage, colTitle) in enumerate(zip(stages, colTitles)):
    rvals = data[f'{stage}_r_values']
    isGrad = data[f'{stage}_is_gradient'].astype(bool)
    lossRat = data[f'{stage}_loss_ratios']
    flipFrac = data[f'{stage}_flip_fracs']

    # Separate gradient vs random
    gradIdx = np.where(isGrad)[0][0]
    randIdx = np.where(~isGrad)[0]

    gradLoss = lossRat[gradIdx]
    gradFlip = flipFrac[gradIdx]
    randLoss = lossRat[randIdx]
    randFlip = flipFrac[randIdx]

    # Median of random
    medLoss = np.median(randLoss, axis=0)
    medFlip = np.median(randFlip, axis=0)

    # Percentiles for shading
    p10Loss, p90Loss = np.percentile(randLoss, [10, 90], axis=0)
    p10Flip, p90Flip = np.percentile(randFlip, [10, 90], axis=0)

    # Find r=1 index for annotations
    r1idx = np.argmin(np.abs(rvals - 1.0))

    # === ROW 0: LOSS RATIO ===
    ax = axes[0, col]

    # Random individual (thin gray)
    for i in randIdx[:8]:
        ax.semilogx(rvals, lossRat[i], color=PALETTE['mid'], lw=0.5,
                    alpha=0.3, zorder=1)

    # Shaded band (10-90th percentile)
    ax.fill_between(rvals, p10Loss, p90Loss, color=PALETTE['blue'],
                    alpha=0.2, zorder=2, label='10-90th percentile')

    # Median random
    ax.semilogx(rvals, medLoss, color=PALETTE['blue'], lw=2,
                label='median random' if col == 0 else None, zorder=4)

    # Gradient (thicker)
    ax.semilogx(rvals, gradLoss, color=PALETTE['red'], lw=2.5,
                label='gradient' if col == 0 else None, zorder=5)

    # r=1 line (prominent green dashed)
    ax.axvline(1.0, color=PALETTE['green'], ls='--', lw=2, zorder=3, alpha=0.8)

    # Annotations at r=1
    gradVal = gradLoss[r1idx]
    medVal = medLoss[r1idx]
    ax.plot(1.0, gradVal, 'o', color=PALETTE['red'], ms=4, zorder=6)
    ax.plot(1.0, medVal, 'o', color=PALETTE['blue'], ms=4, zorder=6)

    # Value labels at r=1 (stagger if close)
    yoffset = 0.4 if abs(gradVal - medVal) < 0.5 else 0
    ax.text(1.2, gradVal + yoffset, f'{gradVal:.2f}', fontsize=7,
            color=PALETTE['red'], va='center', fontweight='bold')
    ax.text(1.2, medVal - yoffset, f'{medVal:.2f}', fontsize=7,
            color=PALETTE['blue'], va='center', fontweight='bold')

    if col == 0:
        ax.set_ylabel('Loss ratio', fontweight='bold')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.95)

    ax.set_ylim(0, 10)
    ax.yaxis.grid(True, alpha=0.25, color=PALETTE['light'])
    ax.set_axisbelow(True)
    ax.set_title(colTitle, fontweight='bold', pad=8)

    # === ROW 1: FLIP FRACTION ===
    ax = axes[1, col]

    # Random individual
    for i in randIdx[:8]:
        ax.semilogx(rvals, flipFrac[i], color=PALETTE['mid'], lw=0.5,
                    alpha=0.3, zorder=1)

    # Shaded band
    ax.fill_between(rvals, p10Flip, p90Flip, color=PALETTE['blue'],
                    alpha=0.2, zorder=2)

    # Median random
    ax.semilogx(rvals, medFlip, color=PALETTE['blue'], lw=2, zorder=4)

    # Gradient (thicker)
    ax.semilogx(rvals, gradFlip, color=PALETTE['red'], lw=2.5, zorder=5)

    # r=1 line (prominent green dashed)
    ax.axvline(1.0, color=PALETTE['green'], ls='--', lw=2, zorder=3, alpha=0.8)

    # Annotations at r=1
    gradVal = gradFlip[r1idx]
    medVal = medFlip[r1idx]
    ax.plot(1.0, gradVal, 'o', color=PALETTE['red'], ms=4, zorder=6)
    ax.plot(1.0, medVal, 'o', color=PALETTE['blue'], ms=4, zorder=6)

    # Value labels at r=1 (stagger if close)
    yoffset = 0.06 if abs(gradVal - medVal) < 0.1 else 0
    ax.text(1.2, gradVal + yoffset, f'{gradVal:.0%}', fontsize=7,
            color=PALETTE['red'], va='center', fontweight='bold')
    ax.text(1.2, medVal - yoffset, f'{medVal:.0%}', fontsize=7,
            color=PALETTE['blue'], va='center', fontweight='bold')

    if col == 0:
        ax.set_ylabel('Flip fraction', fontweight='bold')

    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, alpha=0.25, color=PALETTE['light'])
    ax.set_axisbelow(True)
    ax.set_xlabel(r'$r = \tau / \rho_a$', fontweight='bold')
    ax.set_xlim(0.1, 10)

# Main title with subtitle
fig.suptitle('Transition near r=1 validates the analyticity radius as stability boundary',
             fontsize=11, fontweight='bold', x=0.5, y=0.98, ha='center')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# === SAVE ===
outdir = Path(__file__).resolve().parent / "results"
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "random-sweep.pdf", bbox_inches='tight', dpi=300)
fig.savefig(outdir / "random-sweep.png", bbox_inches='tight', dpi=150)
print(f"Saved to {outdir / 'random-sweep.pdf'}")
plt.show()
