import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for GIF saving
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Tuple, Optional

from generator import generate_items, sample_training_data
from validation import load_config

_CFG = load_config(filename="config_retrocue.yaml")["data"]
from viz.visualizations import _item_colors_from_wheel



def save_samples_3d_gif(
    n_items: int,
    n_samples_per_item: Optional[int] = None,
    loc_std: Optional[float] = None,
    color_std: Optional[float] = None,
    loc_encoding_noise_std: Optional[float] = None,
    color_encoding_noise_std: Optional[float] = None,
    seed: Optional[int] = None,
    items: Optional[List[Tuple[float, float]]] = None,
    save_dir: str = "visualizations",
    n_frames: int = 72,
    rotation_speed: float = 5.0,
    elev: float = 25.0,
    interval: int = 120,
    alpha: float = 0.6,
    marker_size: float = 20.0,
):
    """
    Save a slowly-rotating 3-D scatter plot of training samples as a GIF.

    Mirrors the logic of ``create_gp_surface_3d_gif`` in visualizations.py:
    each frame clears the axes, redraws the scatter at a new azimuth angle,
    and saves all frames with Pillow.

    Axes
    ----
      x = location (°)
      y = color    (°)
      z = weight

    Each item uses a distinct color.  True item positions are marked with a
    star (*) at z = 0 (base plane).

    Parameters
    ----------
    n_items : int
        Number of items.
    n_samples_per_item : int
        Noisy samples per item.
    loc_std, color_std : float
        Location / color noise std (degrees).
    loc_encoding_noise_std, color_encoding_noise_std : float
        Optional encoding-noise std (degrees).
    seed : int, optional
        Random seed.
    items : list of (loc, color) tuples, optional
        Use pre-generated items instead of sampling new ones.
    save_dir : str
        Directory to save the GIF.
    filename : str
        Output filename.
    n_frames : int
        Total animation frames (= full 360 ° / rotation_speed).
    rotation_speed : float
        Degrees of camera azimuth change per frame.
    elev : float
        Camera elevation angle.
    interval : int
        Milliseconds between frames.
    alpha : float
        Point transparency.
    marker_size : float
        Scatter marker size.
    """
    # Fall back to config.yaml defaults for unspecified data parameters
    if n_samples_per_item is None:
        n_samples_per_item = _CFG["n_samples_per_item"]
    if loc_std is None:
        loc_std = _CFG["loc_std"]
    if color_std is None:
        color_std = _CFG["color_std"]
    if loc_encoding_noise_std is None:
        loc_encoding_noise_std = _CFG["loc_encoding_noise_std"]
    if color_encoding_noise_std is None:
        color_encoding_noise_std = _CFG["color_encoding_noise_std"]

    print(f"Creating 3D input-samples GIF  ({n_items} items, {n_samples_per_item} samples/item)…")

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

    # Split data per item once (avoid per-frame re-computation)
    # One colorwheel RGB per item (uses the shared helper from visualizations.py)
    item_rgbs = _item_colors_from_wheel([it[1] for it in items])  # (n_items, 3)

    per_item = []
    for i in range(n_items):
        mask = ids_np == i
        per_item.append({
            "locs":    locs_all[mask],
            "colors":  colors_all[mask],
            "weights": weights_np[mask],
            "color":   item_rgbs[i],
            "true_loc": items[i][0],
            "true_col": items[i][1],
            "label": f"Item {i+1}  loc={items[i][0]:.1f}°  col={items[i][1]:.1f}°",
        })

    z_min = 0.0
    z_max = float(weights_np.max()) * 1.05

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    fig.tight_layout()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(frame):
        ax.clear()

        ax.set_xlabel("Location (°)", labelpad=6)
        ax.set_ylabel("Color (°)",    labelpad=6)
        ax.set_zlabel("Weight",       labelpad=6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_zlim(z_min, z_max)

        # Slowly rotate camera (same pattern as create_gp_surface_3d_gif)
        azim = 45 + frame * rotation_speed
        ax.view_init(elev=elev, azim=azim)

        for d in per_item:
            # Sample cloud
            ax.scatter(
                d["locs"], d["colors"], d["weights"],
                color=d["color"], alpha=alpha, s=marker_size,
                depthshade=True, label=d["label"],
            )
            # True position marker at base plane
            ax.scatter(
                [d["true_loc"]], [d["true_col"]], [z_min],
                color=d["color"], marker="*", s=200,
                edgecolors="white", linewidths=0.8,
                depthshade=False,
            )

        return fig,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"input_samples_3d_{n_items}.gif")
    anim.save(out_path, dpi=100, writer="pillow")
    plt.close()
    print(f"  Saved → {out_path}")


# ── quick demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for n in [4]:
        save_samples_3d_gif(n_items=n,seed=42)
