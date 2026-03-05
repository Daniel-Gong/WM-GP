import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional

from generator import generate_items, sample_training_data


# 10 distinct plotly-friendly colors
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_samples_3d(
    n_items: int,
    n_samples_per_item: int = 50,
    loc_std: float = 20.0,
    color_std: float = 20.0,
    loc_encoding_noise_std: float = 0.0,
    color_encoding_noise_std: float = 0.0,
    seed: Optional[int] = None,
    items: Optional[List[Tuple[float, float]]] = None,
    opacity: float = 0.5,
    marker_size: int = 4,
):
    """
    Visualize training samples for N items in an interactive 3-D scatter plot
    (opens in browser, fully rotatable / zoomable with the mouse).

    Axes
    ----
      x = location (°)
      y = color    (°)
      z = weight

    Each item uses a distinct color.  True item positions are shown as large
    stars at the peak weight of their cluster.

    Parameters
    ----------
    n_items : int
        Number of items to generate / visualise.
    n_samples_per_item : int
        Noisy samples drawn per item.
    loc_std, color_std : float
        Location / color noise std (degrees).
    loc_encoding_noise_std, color_encoding_noise_std : float
        Optional encoding-noise std (degrees).
    seed : int, optional
        Random seed for reproducibility.
    items : list of (loc, color) tuples, optional
        Supply your own items instead of generating new ones.
    opacity : float
        Point transparency (0–1).
    marker_size : int
        Scatter marker size (px).
    """
    if items is None:
        items = generate_items(n_items, seed=seed)

    samples, weights, item_ids = sample_training_data(
        items,
        n_samples_per_item=n_samples_per_item,
        loc_std=loc_std,
        color_std=color_std,
        loc_encoding_noise_std=loc_encoding_noise_std,
        color_encoding_noise_std=color_encoding_noise_std,
    )

    samples_np = samples.cpu().numpy()   # (N*S, 2)
    weights_np = weights.cpu().numpy()   # (N*S,)
    ids_np     = item_ids.cpu().numpy()  # (N*S,)

    locs_all   = samples_np[:, 0]
    colors_all = samples_np[:, 1]

    traces = []

    for i in range(n_items):
        col  = _PALETTE[i % len(_PALETTE)]
        mask = ids_np == i

        # ── sample cloud ────────────────────────────────────────────────────
        traces.append(go.Scatter3d(
            x=locs_all[mask].tolist(),
            y=colors_all[mask].tolist(),
            z=weights_np[mask].tolist(),
            mode="markers",
            name=f"Item {i+1}  loc={items[i][0]:.1f}°  col={items[i][1]:.1f}°",
            marker=dict(size=marker_size, color=col, opacity=opacity),
            hovertemplate=(
                "loc: %{x:.1f}°<br>"
                "color: %{y:.1f}°<br>"
                "weight: %{z:.3f}<extra></extra>"
            ),
        ))

        # ── true-position star ──────────────────────────────────────────────
        z_star = 0.0
        traces.append(go.Scatter3d(
            x=[items[i][0]],
            y=[items[i][1]],
            z=[z_star],
            mode="markers",
            name=f"★ Item {i+1} (true)",
            showlegend=False,
            marker=dict(
                size=8,
                symbol="circle",
                color=col,
                line=dict(color="black", width=1),
            ),
            hovertemplate=(
                f"<b>Item {i+1} true position</b><br>"
                "loc: %{x:.1f}°<br>"
                "color: %{y:.1f}°<extra></extra>"
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(
                f"Training samples — {n_items} item(s), "
                f"{n_samples_per_item} samples/item<br>"
                f"<sup>loc_std={loc_std}°  color_std={color_std}°</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Location (°)", range=[-180, 180]),
            yaxis=dict(title="Color (°)",    range=[-180, 180]),
            zaxis=dict(title="Weight"),
        ),
        legend=dict(x=1.0, y=0.9),
        margin=dict(l=0, r=0, b=0, t=60),
        width=900,
        height=700,
    )

    fig.show()   # opens in default browser — fully interactive


# ── quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_samples_3d(
        n_items=6,
        n_samples_per_item=1000,
        loc_std=20.0,
        color_std=20.0
    )
