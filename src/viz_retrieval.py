"""
viz_retrieval.py
----------------
Visualise the GP-based memory retrieval mechanism in two side-by-side panels.

Panel 1 – GP surface heatmap (via vis.plot_gp_surface_2d logic, inlined so it
           returns a figure rather than saving).  A vertical dashed line marks
           the cued location on the x-axis.

Panel 2 – The GP predictive mean profile used by `retrieve_color` (simulation.py
           line 35: `means = pred.mean`), shown as a function of candidate color
           at the queried location.  The argmax position (= retrieved color) is
           annotated with a vertical dotted line and an arrow.

Usage
-----
    python viz_retrieval.py        # runs a demo with random items
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gpytorch
from typing import List, Tuple, Optional

from generator import generate_items
from simulation import run_simulation_trial
import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Main visualisation function
# ──────────────────────────────────────────────────────────────────────────────

def plot_retrieval_mechanism(
    model,
    likelihood,
    items: List[Tuple[float, float]],
    cue_item_idx: int = 0,
    n_color_samples: int = 360,
    surface_res: int = 60,
    epoch_label: str = "Final",
    title: str = "GP Memory Retrieval",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a two-panel figure illustrating the GP memory retrieval mechanism.

    Parameters
    ----------
    model : WorkingMemoryGP
        Trained GP model (in eval mode after this call).
    likelihood : GaussianLikelihood
        Corresponding likelihood.
    items : list of (loc, color) tuples
        Ground-truth items (in degrees, range [-180, 180]).
    cue_item_idx : int
        Index of the cued item; its location is used for the retrieval profile
        (Panel 2) and a vertical cue line is drawn in Panel 1.
    n_color_samples : int
        Number of candidate colors evaluated in the retrieval profile.
    surface_res : int
        Grid resolution of the 2-D surface heatmap (Panel 1).
    epoch_label : str
        Label appended to the figure suptitle.
    save_path : str, optional
        If given, save the figure here; otherwise show interactively.

    Returns
    -------
    fig : matplotlib.Figure
    """
    model.eval()
    likelihood.eval()

    cue_loc, cue_color = items[cue_item_idx]

    # ── build 2-D surface (Panel 1) ──────────────────────────────────────────
    locs   = torch.linspace(-180.0, 180.0, surface_res, device=device)
    colors = torch.linspace(-180.0, 180.0, surface_res, device=device)
    L, C   = torch.meshgrid(locs, colors, indexing="ij")
    grid   = torch.stack([L.flatten(), C.flatten()], dim=-1)

    with torch.no_grad():
        surf_preds = likelihood(model(grid))
        mean_surface = surf_preds.mean.view(surface_res, surface_res).cpu().numpy()

    # ── build 1-D retrieval profile (Panel 2) ────────────────────────────────
    color_samples = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    query_points  = torch.stack(
        [torch.tensor([cue_loc, c], device=device) for c in color_samples]
    )

    with torch.no_grad():
        pred  = model(query_points)       # same call as simulation.py L34
        means = pred.mean                 # simulation.py L35
        best_idx    = torch.argmax(means)
        best_color  = color_samples[best_idx].item()
        means_np    = means.cpu().numpy()

    color_np = color_samples.cpu().numpy()

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.suptitle(
        f"{title}  |  Epoch: {epoch_label}  |  Cued item #{cue_item_idx + 1}"
        f"  (loc={cue_loc:.1f}°, true color={cue_color:.1f}°)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1 – GP predictive mean surface
    # ════════════════════════════════════════════════════════════════════════
    ax1 = axes[0]

    im = ax1.imshow(
        mean_surface.T,
        extent=[-180.0, 180.0, -180.0, 180.0],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cbar.set_label("Predictive Mean (memory strength)", fontsize=9)

    # True item positions
    true_locs  = [i[0] for i in items]
    true_cols  = [i[1] for i in items]
    ax1.scatter(
        true_locs, true_cols,
        c="red", marker="*", s=250, zorder=5,
        label="Items", edgecolors="white", linewidths=0.5,
    )

    # Highlight cued item with a distinct marker
    ax1.scatter(
        [cue_loc], [cue_color],
        c="yellow", marker="*", s=350, zorder=6,
        edgecolors="black", linewidths=0.8, label=f"Cued item #{cue_item_idx + 1}",
    )

    # Inducing points
    if hasattr(model, "variational_strategy"):
        ind_pts = model.variational_strategy.inducing_points.detach().cpu().numpy()
        ax1.scatter(
            ind_pts[:, 0], ind_pts[:, 1],
            c="white", marker=".", s=40, alpha=0.4, zorder=4, label="Inducing pts",
        )

    # ── vertical dotted cue line ──────────────────────────────────────────────
    ax1.axvline(
        x=cue_loc,
        color="gold", linestyle=":", linewidth=2.2, zorder=7,
        label=f"Cue loc {cue_loc:.1f}°",
    )

    ax1.set_title("GP Predictive Mean Surface", fontsize=11)
    ax1.set_xlabel("Location (°)", fontsize=10)
    ax1.set_ylabel("Color (°)", fontsize=10)
    ax1.set_xlim([-180.0, 180.0])
    ax1.set_ylim([-180.0, 180.0])
    ax1.legend(fontsize=8, loc="upper right")

    # ════════════════════════════════════════════════════════════════════════
    # Panel 2 – Retrieval profile: GP mean along color axis at cue location
    # ════════════════════════════════════════════════════════════════════════
    ax2 = axes[1]

    ax2.plot(
        color_np, means_np,
        color="#2196F3", linewidth=2, label="GP posterior mean",
    )
    ax2.fill_between(color_np, means_np, alpha=0.12, color="#2196F3")

    # True color vertical line
    ax2.axvline(
        x=cue_color,
        color="red", linestyle="--", linewidth=1.8, zorder=5,
        label=f"True color {cue_color:.1f}°",
    )

    # Argmax (retrieved color) annotation
    peak_val = means_np[best_idx.item()]
    ax2.axvline(
        x=best_color,
        color="gold", linestyle=":", linewidth=2.2, zorder=6,
        label=f"Retrieved (argmax) {best_color:.1f}°",
    )

    # Arrow annotation at the peak
    arrow_yoffset = (means_np.max() - means_np.min()) * 0.18
    ax2.annotate(
        f"argmax\n{best_color:.1f}°",
        xy=(best_color, peak_val),
        xytext=(best_color + 25 if best_color < 120 else best_color - 25,
                peak_val + arrow_yoffset),
        fontsize=9, color="darkgoldenrod", fontweight="bold",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            color="darkgoldenrod",
            lw=1.5,
        ),
    )

    # Mark the peak with a dot
    ax2.scatter(
        [best_color], [peak_val],
        color="gold", s=80, zorder=7, edgecolors="black", linewidths=0.8,
    )

    ax2.set_title(
        f"Retrieval profile at loc = {cue_loc:.1f}°\n"
        r"$\hat{c} = \arg\max_c\; \mu_{GP}(\ell_{cue},\, c)$",
        fontsize=11,
    )
    ax2.set_xlabel("Candidate color (°)", fontsize=10)
    ax2.set_ylabel("GP posterior mean $\\mu$", fontsize=10)
    ax2.set_xlim([-180.0, 180.0])
    ax2.set_xticks([-180, -90, 0, 90, 180])
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Annotate error
    error = best_color - cue_color
    # Wrap to [-180, 180]
    error = (error + 180) % 360 - 180
    ax2.text(
        0.02, 0.97,
        f"Error = {error:+.1f}°",
        transform=ax2.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading config …")
    cfg = _load_config()

    n_items = 3
    print(f"Generating {n_items} items …")
    items = generate_items(n_items, seed=42)
    print(f"  Items: {items}")

    print("Running simulation trial (encoding only) …")
    model, likelihood, history = run_simulation_trial(
        items,
        config=cfg,
        maintenance_epochs=0,    # skip maintenance for speed
        track_visuals=False,
    )

    print("Plotting retrieval mechanism …")
    fig = plot_retrieval_mechanism(
        model,
        likelihood,
        items,
        cue_item_idx=0,
        epoch_label="Final",
        title="GP Memory Retrieval Demo",
    )
