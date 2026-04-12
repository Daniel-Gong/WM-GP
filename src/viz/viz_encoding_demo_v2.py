"""
viz_encoding_demo_v2.py
========================
Encoding-process demonstration — 3 panels, each saved as a separate PDF.

Panel 1 – 3D grid axes + true item positions (stars at z = 0)
Panel 2 – Panel 1 + ALL sample points projected onto the floor (z = 0)
Panel 3 – Panel 2 + inducing points & GP surface AFTER encoding

Usage
-----
    python viz_encoding_demo_v2.py

Output PDFs are written to  ../visualizations/encoding_demo/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from generator import generate_items, sample_training_data
from gp_model import WorkingMemoryGP
from validation import load_config
from viz.visualizations import _item_colors_from_wheel

# ── reproducibility ────────────────────────────────────────────────────────────
SEED        = 42
N_ITEMS     = 4
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "visualizations", "encoding_demo")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── shared cosmetics ───────────────────────────────────────────────────────────
ELEV        = 28.0
AZIM        = -55.0
ALPHA_CLOUD = 0.45
MARKER_SIZE = 8           # scatter s= for the sample cloud
STAR_SIZE   = 250         # scatter s= for the true-item stars
FIG_SIZE    = (7, 6)

device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
cfg  = load_config(filename="config_retrocue.yaml")
dcfg = cfg["data"]
mcfg = cfg["model"]

items     = generate_items(N_ITEMS, seed=SEED)
item_rgbs = _item_colors_from_wheel([it[1] for it in items])

samples, weights, item_ids = sample_training_data(
    items,
    n_samples_per_item       = dcfg["n_samples_per_item"],
    loc_std                  = dcfg["loc_std"],
    color_std                = dcfg["color_std"],
    loc_encoding_noise_std   = dcfg["loc_encoding_noise_std"],
    color_encoding_noise_std = dcfg["color_encoding_noise_std"],
)

samples_np = samples.cpu().numpy()
weights_np = weights.cpu().numpy()
ids_np     = item_ids.cpu().numpy()

locs_all   = samples_np[:, 0]
colors_all = samples_np[:, 1]

# Per-item data (same split as before)
per_item = []
for i in range(N_ITEMS):
    mask = ids_np == i
    per_item.append(dict(
        locs     = locs_all[mask],
        colors   = colors_all[mask],
        weights  = weights_np[mask],
        color    = item_rgbs[i],
        true_loc = items[i][0],
        true_col = items[i][1],
    ))

# ══════════════════════════════════════════════════════════════════════════════
# Train GP (posterior for Panel 3)
# ══════════════════════════════════════════════════════════════════════════════
SURF_RES = 60

locs_g   = np.linspace(-180, 180, SURF_RES)
colors_g = np.linspace(-180, 180, SURF_RES)
L_g, C_g = np.meshgrid(locs_g, colors_g, indexing="ij")
grid_t   = torch.tensor(
    np.column_stack([L_g.ravel(), C_g.ravel()]), dtype=torch.float32
)

model_post = WorkingMemoryGP(
    inducing_grid_size       = mcfg["inducing_grid_size"],
    loc_lengthscale          = mcfg["loc_lengthscale"],
    color_lengthscale        = mcfg["color_lengthscale"],
    learn_inducing_locations = mcfg["learn_inducing_locations"],
).to(device)
likelihood_post = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
).to(device)
likelihood_post.noise = cfg["likelihood"]["noise_variance"]
likelihood_post.noise_covar.raw_noise.requires_grad_(False)

optimizer = torch.optim.Adam(model_post.parameters(), lr=cfg["training"]["encoding_lr"])
mll       = gpytorch.mlls.VariationalELBO(likelihood_post, model_post, num_data=len(samples))

model_post.train(); likelihood_post.train()
print("Training GP …")
for _ in range(cfg["training"]["encoding_epochs"]):
    optimizer.zero_grad()
    loss = -mll(model_post(samples), weights)
    loss.backward()
    optimizer.step()

model_post.eval(); likelihood_post.eval()
with torch.no_grad():
    post_mean = likelihood_post(model_post(grid_t)).mean.numpy().reshape(SURF_RES, SURF_RES)
    ind_pts_post  = model_post.variational_strategy.inducing_points.detach().cpu().numpy()
    ind_vals_post = likelihood_post(model_post(
        torch.tensor(ind_pts_post, dtype=torch.float32)
    )).mean.detach().numpy()

Z_MAX_POST = float(post_mean.max()) * 1.15

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _setup_ax(ax, z_max):
    ax.set_xlabel("Location (°)", labelpad=6, fontsize=10)
    ax.set_ylabel("Color (°)",    labelpad=6, fontsize=10)
    ax.set_zlabel("Activation",   labelpad=6, fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_zlim(0, z_max)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(axis='both', which='major', labelsize=7)


def _draw_items_at_base(ax):
    for d in per_item:
        ax.scatter(
            [d["true_loc"]], [d["true_col"]], [0.0],
            color=d["color"], marker="*", s=STAR_SIZE,
            edgecolors="white", linewidths=0.8, depthshade=False, zorder=6,
        )


def _draw_floor_samples(ax):
    """All sample points projected to z = 0 (the floor / xy plane)."""
    for d in per_item:
        ax.scatter(
            d["locs"], d["colors"], np.zeros_like(d["locs"]),
            color=d["color"], alpha=ALPHA_CLOUD, s=MARKER_SIZE,
            depthshade=False,
        )


def _new_fig():
    fig = plt.figure(figsize=FIG_SIZE)
    ax  = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    return fig, ax


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — 3D grid + true items only
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 1 …")
fig1, ax1 = _new_fig()
_setup_ax(ax1, Z_MAX_POST)
_draw_items_at_base(ax1)
ax1.set_title("Panel 1 — True Items", fontsize=11, pad=8)
_save(fig1, "v2_panel1_true_items.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Panel 1 + sample cloud projected on the floor (z = 0)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 2 …")
fig2, ax2 = _new_fig()
_setup_ax(ax2, Z_MAX_POST)
_draw_items_at_base(ax2)
_draw_floor_samples(ax2)
ax2.set_title("Panel 2 — True Items + Samples (floor)", fontsize=11, pad=8)
_save(fig2, "v2_panel2_floor_samples.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Panel 2 + inducing points & posterior surface after encoding
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 3 …")
fig3, ax3 = _new_fig()
_setup_ax(ax3, Z_MAX_POST)
_draw_items_at_base(ax3)
_draw_floor_samples(ax3)

ax3.plot_surface(
    L_g, C_g, post_mean,
    cmap="viridis", edgecolor="none", alpha=0.55,
    vmin=0, vmax=Z_MAX_POST,
)
ax3.scatter(
    ind_pts_post[:, 0], ind_pts_post[:, 1], ind_vals_post,
    color="red", edgecolors="white", s=50, linewidths=0.5,
    depthshade=False, zorder=8, label="Inducing pts (posterior)",
)
ax3.set_title("Panel 3 — + Inducing Points (Posterior)", fontsize=11, pad=8)
ax3.legend(fontsize=7, loc="upper right")
_save(fig3, "v2_panel3_posterior.pdf")

print("\nDone!  All 3 PDFs saved to:", SAVE_DIR)
