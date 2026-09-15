"""
Shared ICRA figure style for all visualization scripts.
Colorblind-friendly palette, English labels, PDF/PNG output.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (colorblind-friendly, fixed per method) ──────────────
COLOR_GREEDY = "#7f7f7f"       # gray
COLOR_OLD_WM = "#d95f02"       # orange
COLOR_CONG_WM = "#1b9e77"      # teal-green
COLOR_ORACLE = "#7570b3"       # purple
COLOR_ACCENT = "#e7298a"       # magenta (spare)

METHOD_COLORS = {
    "Greedy": COLOR_GREEDY,
    "Old-cost WM": COLOR_OLD_WM,
    "Cong-cost WM": COLOR_CONG_WM,
    "Oracle": COLOR_ORACLE,
}

METHOD_HATCHES = {
    "Greedy": "",
    "Old-cost WM": "",
    "Cong-cost WM": "",
    "Oracle": "//",
}

# ── Global matplotlib RC ─────────────────────────────────────────
def apply_icra_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── Save helper (PDF + PNG) ─────────────────────────────────────
def save_fig(fig, path_stem, dpi_png=300):
    """Save as PDF and PNG. `path_stem` has no extension."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=dpi_png)
    plt.close(fig)


# ── Standard error / 95% CI helper ──────────────────────────────
def mean_ci95(vals):
    """Return (mean, ci95_half_width) from a list of numbers."""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    m = arr.mean()
    if n < 2:
        return m, 0.0
    se = arr.std(ddof=1) / np.sqrt(n)
    return m, 1.96 * se


def mean_se(vals):
    """Return (mean, standard_error)."""
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    m = arr.mean()
    if n < 2:
        return m, 0.0
    return m, arr.std(ddof=1) / np.sqrt(n)


# ── Subplot label helper ─────────────────────────────────────────
def label_subplot(ax, letter, x=-0.12, y=1.08):
    """Add (a), (b), ... label to subplot."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
