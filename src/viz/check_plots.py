import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
try:
    matplotlib.use("macosx")   # macOS native — supports mouse rotation
except Exception:
    matplotlib.use("TkAgg")    # fallback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Tuple, Optional

from generator import generate_items, sample_training_data


def plot_samples_3d(
    n_items: int,
    n_samples_per_item: int = 50,
    loc_std: float = 20.0,
    color_std: float = 20.0,
    loc_encoding_noise_std: float = 0.0,
    color_encoding_noise_std: float = 0.0,
    seed: Optional[int] = None,
    items: Optional[List[Tuple[float, float]]] = None,
    alpha: float = 0.4,
    marker_size: float = 15.0,
):
    """
    Visualize training samples for N items in a 3D scatter plot.

    The three axes are:
      x = location (degrees)
      y = color    (degrees)
      z = weight   (computed inside sample_training_data)

    Each item gets its own color.  The true item positions are marked
    with a large star (★) on the x-y plane (z = max weight).

    Parameters
    ----------
    n_items : int
        Number of items to generate / visualize.
    n_samples_per_item : int
        Number of noisy samples per item.
    loc_std : float
        Location noise std (degrees).
    color_std : float
        Color noise std (degrees).
    loc_encoding_noise_std : float
        Optional encoding noise std for location.
    color_encoding_noise_std : float
        Optional encoding noise std for color.
    seed : int, optional
        Random seed for reproducibility.
    items : list of (loc, color) tuples, optional
        If provided, use these items instead of generating new ones.
    alpha : float
        Transparency of scatter points.
    marker_size : float
        Size of scatter markers.
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

    # Convert to numpy for plotting
    samples_np = samples.cpu().numpy()   # (N*n_samples, 2)
    weights_np = weights.cpu().numpy()   # (N*n_samples,)
    ids_np     = item_ids.cpu().numpy()  # (N*n_samples,)

    locs   = samples_np[:, 0]
    colors = samples_np[:, 1]

    # Color palette — one distinct color per item
    cmap = plt.get_cmap("tab10")
    item_colors = [cmap(i % 10) for i in range(n_items)]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    for i in range(n_items):
        mask = ids_np == i
        ax.scatter(
            locs[mask],
            colors[mask],
            weights_np[mask],
            color=item_colors[i],
            alpha=alpha,
            s=marker_size,
            label=f"Item {i+1}  loc={items[i][0]:.1f}°  col={items[i][1]:.1f}°",
        )

        # Mark true item position as a star at z = max weight of that item
        true_loc, true_col = items[i]
        z_star = weights_np[mask].max() if mask.any() else 1.0
        ax.scatter(
            [true_loc],
            [true_col],
            [z_star],
            color=item_colors[i],
            marker="*",
            s=250,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xlabel("Location (°)", labelpad=8)
    ax.set_ylabel("Color (°)",    labelpad=8)
    ax.set_zlabel("Weight",       labelpad=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_title(
        f"Training samples — {n_items} item(s), "
        f"{n_samples_per_item} samples/item\n"
        f"loc_std={loc_std}°, color_std={color_std}°",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.show()


# ── quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_samples_3d(
        n_items=3,
        n_samples_per_item=80,
        loc_std=20.0,
        color_std=20.0,
        seed=42,
    )
