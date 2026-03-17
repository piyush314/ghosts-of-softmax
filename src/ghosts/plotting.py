"""Shared plotting helpers for public figures and tutorials."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


PALETTE = {
    "red": "#E3120B",
    "blue": "#006BA2",
    "teal": "#00A5A5",
    "gold": "#F4A100",
    "purple": "#6F2DA8",
    "green": "#00843D",
    "dark_gray": "#3D3D3D",
    "mid_gray": "#767676",
    "light_gray": "#D0D0D0",
    # Backward-compatible aliases used in older experiment code.
    "dark": "#3D3D3D",
    "mid": "#767676",
    "light": "#D0D0D0",
}


def apply_plot_style(
    *,
    font_size: int = 10,
    title_size: int = 14,
    label_size: int = 10,
    tick_size: int = 9,
    legend_size: int = 9,
) -> None:
    """Apply a consistent publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.titleweight": "bold",
            "axes.labelsize": label_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["dark_gray"],
            "axes.linewidth": 0.8,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "xtick.color": PALETTE["dark_gray"],
            "ytick.color": PALETTE["dark_gray"],
            "grid.color": PALETTE["light_gray"],
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": legend_size,
        }
    )


def add_subtitle(ax, subtitle: str, *, fontsize: int = 10) -> None:
    """Add a left-aligned subtitle under the title."""
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=PALETTE["mid_gray"],
        ha="left",
        va="bottom",
    )


def finish_figure(fig, *, rect: Sequence[float] | None = None) -> None:
    """Reserve title space when needed and tighten layout."""
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)


def format_percent_axis(
    ax,
    *,
    axis: str = "y",
    xmax: float = 1.0,
    decimals: int = 0,
) -> None:
    """Apply percent formatting to a proportion axis."""
    formatter = ticker.PercentFormatter(xmax=xmax, decimals=decimals)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def add_end_labels(
    ax,
    x_values: Sequence[float],
    specs: Iterable[tuple[float, str, str, str | None]],
    *,
    x_pad_frac: float = 0.05,
    fontsize: int = 9,
) -> None:
    """Add right-edge labels for multi-series line charts."""
    xmin, xmax = ax.get_xlim()
    if xmax > 0:
        x_text = xmax * (1.0 + x_pad_frac)
        ax.set_xlim(xmin, xmax * (1.0 + 2.3 * x_pad_frac))
    else:
        span = xmax - xmin
        x_text = xmax + span * x_pad_frac
        ax.set_xlim(xmin, xmax + span * 2.3 * x_pad_frac)

    for y_value, label, color, weight in specs:
        ax.text(
            x_text,
            y_value,
            label,
            color=color,
            fontsize=fontsize,
            ha="left",
            va="center",
            fontweight=weight or "normal",
        )
