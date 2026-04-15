"""
Figure 1: Conceptual schematic of the MemGP framework.

Panel A — Pipeline: Stimulus → Encoding → Maintenance → Retrieval
Panel B — 2D GP surface example (N=4 items)
Panel C — Attention gain function
Panel D — Inducing-point grid capacity constraint
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib import patheffects

_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PAPER_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC)


def _draw_pipeline(ax):
    """Panel A: high-level pipeline diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    boxes = [
        (0.3,  "Stimulus\nArray",        "#E8F5E9", "#2E7D32"),
        (2.3,  "Encoding\n(ELBO fit)",    "#E3F2FD", "#1565C0"),
        (4.3,  "Maintenance\n(Self-rehearsal\n+ Attention)",  "#FFF3E0", "#E65100"),
        (6.3,  "Retrieval\n(Posterior\nevaluation)", "#F3E5F5", "#6A1B9A"),
        (8.3,  "Response\n(Color report)", "#FFEBEE", "#C62828"),
    ]

    box_w, box_h = 1.7, 1.6
    y_center = 1.25

    for x, label, facecolor, edgecolor in boxes:
        bbox = FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.0,
        )
        ax.add_patch(bbox)
        ax.text(
            x + box_w / 2, y_center, label,
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=edgecolor,
        )

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w + 0.02
        x_end = boxes[i + 1][0] - 0.02
        ax.annotate(
            "", xy=(x_end, y_center), xytext=(x_start, y_center),
            arrowprops=dict(arrowstyle="-|>", color="grey", lw=2.0),
        )

    math_labels = [
        (boxes[1][0] + box_w / 2, y_center - box_h / 2 - 0.25,
         r"$\mathcal{L}_{\mathrm{enc}} = \mathbb{E}_q[\sum w_i \log p(y_i|f)] - \mathrm{KL}$",
         "#1565C0"),
        (boxes[2][0] + box_w / 2, y_center - box_h / 2 - 0.25,
         r"$\mathcal{L}_{\mathrm{maint}} = -\sum a_g \mathbb{E}_q[\log p(\tilde{y}_g|f)] + \beta \mathrm{KL}$",
         "#E65100"),
        (boxes[3][0] + box_w / 2, y_center - box_h / 2 - 0.25,
         r"$\hat{c} = \arg\max_c \mu(\ell_{\mathrm{probe}}, c)$",
         "#6A1B9A"),
    ]
    for x, y, txt, color in math_labels:
        ax.text(x, y, txt, ha="center", va="top", fontsize=6.5,
                color=color, style="italic")


def _draw_gp_surface(ax):
    """Panel B: synthetic 2D GP surface with 4 item peaks."""
    items = [(-90, -60), (-30, 80), (60, -120), (120, 30)]

    locs = np.linspace(-180, 180, 200)
    cols = np.linspace(-180, 180, 200)
    L, C = np.meshgrid(locs, cols, indexing="ij")
    surface = np.zeros_like(L)

    sigma_l, sigma_c = 25.0, 25.0
    for loc, col in items:
        dl = np.minimum(np.abs(L - loc), 360 - np.abs(L - loc))
        dc = np.minimum(np.abs(C - col), 360 - np.abs(C - col))
        surface += np.exp(-0.5 * (dl / sigma_l) ** 2 - 0.5 * (dc / sigma_c) ** 2)

    im = ax.imshow(
        surface.T, extent=[-180, 180, -180, 180],
        origin="lower", cmap="viridis", aspect="auto",
    )
    for loc, col in items:
        ax.scatter(loc, col, c="white", marker="*", s=120,
                   edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("Location (°)", fontsize=8)
    ax.set_ylabel("Color (°)", fontsize=8)
    ax.set_title("GP posterior surface ($N=4$)", fontsize=9, fontweight="bold")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\mu(\\ell, c)$")


def _draw_attention_gain(ax):
    """Panel C: spatial attention gain function."""
    locs = np.linspace(-180, 180, 500)
    cued_loc = 0.0
    sigma_s = 20.0

    for gain, color, ls in [(2, "#4CAF50", "--"), (5, "#FF9800", "-"), (20, "#F44336", "-")]:
        dist = np.minimum(np.abs(locs - cued_loc), 360 - np.abs(locs - cued_loc))
        gaussian = np.exp(-0.5 * (dist / sigma_s) ** 2)
        weights = 1.0 + (gain - 1.0) * gaussian
        ax.plot(locs, weights, color=color, linewidth=2, linestyle=ls,
                label=f"$G={gain}$")

    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Location (°)", fontsize=8)
    ax.set_ylabel("Attention weight $a(\\ell)$", fontsize=8)
    ax.set_title("Attention gain at cued location", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=7)
    ax.text(5, 1.3, "cued\nloc", ha="left", va="bottom", fontsize=6.5, color="grey")


def _draw_inducing_grid(ax):
    """Panel D: inducing point grid and capacity interpretation."""
    grid_sizes = [6, 10, 16]
    colors_g = ["#E53935", "#1E88E5", "#43A047"]
    items = [(-90, -60), (-30, 80), (60, -120), (120, 30)]

    ax.set_xlim(-190, 190)
    ax.set_ylim(-190, 190)

    gs = grid_sizes[1]
    half_sp = 180.0 / gs
    grid_1d = np.linspace(-180 + half_sp, 180 - half_sp, gs)
    Lg, Cg = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    ax.scatter(Lg.ravel(), Cg.ravel(), c=colors_g[1], s=12, alpha=0.6,
              marker="s", label=f"$M={gs}^2={gs**2}$")

    for loc, col in items:
        ax.scatter(loc, col, c="white", marker="*", s=120,
                   edgecolors="black", linewidths=0.5, zorder=5)

    ax.set_xlabel("Location (°)", fontsize=8)
    ax.set_ylabel("Color (°)", fontsize=8)
    ax.set_title("Inducing points ($\\mathbf{Z}$): finite capacity", fontsize=9,
                 fontweight="bold")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_facecolor("#f5f5f5")
    ax.set_aspect("equal")


def create_figure1(save_path=None):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.3],
                           hspace=0.38, wspace=0.35)

    ax_pipeline = fig.add_subplot(gs[0, :])
    ax_surface = fig.add_subplot(gs[1, 0])
    ax_attention = fig.add_subplot(gs[1, 1])
    ax_inducing = fig.add_subplot(gs[1, 2])

    ax_pipeline.text(-0.02, 1.05, "A", transform=ax_pipeline.transAxes,
                     fontsize=16, fontweight="bold", va="top")
    ax_surface.text(-0.15, 1.05, "B", transform=ax_surface.transAxes,
                    fontsize=16, fontweight="bold", va="top")
    ax_attention.text(-0.15, 1.05, "C", transform=ax_attention.transAxes,
                      fontsize=16, fontweight="bold", va="top")
    ax_inducing.text(-0.15, 1.05, "D", transform=ax_inducing.transAxes,
                     fontsize=16, fontweight="bold", va="top")

    _draw_pipeline(ax_pipeline)
    _draw_gp_surface(ax_surface)
    _draw_attention_gain(ax_attention)
    _draw_inducing_grid(ax_inducing)

    if save_path is None:
        save_path = os.path.join(_PAPER_DIR, "fig1_schematic.pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved to {save_path}")


if __name__ == "__main__":
    create_figure1()
