"""
viz_encoding_demo.py
====================
Encoding-process demonstration — 4 panels, each saved as a separate PDF.

Panel 1 – 3D grid axes + true item positions (stars at z = 0)
Panel 2 – Panel 1 + noisy sample cloud (coloured scatter)
Panel 3 – Panel 2 + inducing points sitting on the *prior* GP surface
           (flat, before any training)
Panel 4 – Panel 2 + inducing points and GP surface *after encoding*
           (trained / posterior surface elevated near encoded items)

Usage
-----
    python viz_encoding_demo.py

Output PDFs are written to  ../visualizations/encoding_demo/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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
from visualizations import _item_colors_from_wheel

# ── reproducibility ────────────────────────────────────────────────────────────
SEED        = 42
N_ITEMS     = 4
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "visualizations", "encoding_demo")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── shared cosmetics ───────────────────────────────────────────────────────────
ELEV        = 28.0
AZIM        = -55.0
ALPHA_CLOUD = 0.55
MARKER_SIZE = 12          # scatter s= for the sample cloud
STAR_SIZE   = 250         # scatter s= for the true-item stars
FIG_SIZE    = (7, 6)

device = torch.device("cpu")   # keep demo lightweight

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Generate data once so all panels share the same trial
# ══════════════════════════════════════════════════════════════════════════════
cfg  = load_config(filename="config_retrocue.yaml")
dcfg = cfg["data"]
mcfg = cfg["model"]

items  = generate_items(N_ITEMS, seed=SEED)
item_rgbs = _item_colors_from_wheel([it[1] for it in items])   # (N, 3)

samples, weights, item_ids = sample_training_data(
    items,
    n_samples_per_item = dcfg["n_samples_per_item"],
    loc_std            = dcfg["loc_std"],
    color_std          = dcfg["color_std"],
    loc_encoding_noise_std  = dcfg["loc_encoding_noise_std"],
    color_encoding_noise_std= dcfg["color_encoding_noise_std"],
)

samples_np  = samples.cpu().numpy()   # (N*S, 2)
weights_np  = weights.cpu().numpy()   # (N*S,)
ids_np      = item_ids.cpu().numpy()  # (N*S,)

locs_all   = samples_np[:, 0]
colors_all = samples_np[:, 1]

# Per-item split (reused across panels)
per_item = []
for i in range(N_ITEMS):
    mask = ids_np == i
    per_item.append(dict(
        locs    = locs_all[mask],
        colors  = colors_all[mask],
        weights = weights_np[mask],
        color   = item_rgbs[i],
        true_loc = items[i][0],
        true_col = items[i][1],
        label   = f"Item {i+1}  loc={items[i][0]:.0f}°  col={items[i][1]:.0f}°",
    ))

Z_MIN  = 0.0
Z_MAX  = float(weights_np.max()) * 1.15   # shared z-scale (panels 1-3)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Build GP surface grids
# ══════════════════════════════════════════════════════════════════════════════
SURF_RES = 60   # coarser grid = faster render; increase for final quality

locs_g   = np.linspace(-180, 180, SURF_RES)
colors_g = np.linspace(-180, 180, SURF_RES)
L_g, C_g = np.meshgrid(locs_g, colors_g, indexing="ij")   # (res, res)
grid_t   = torch.tensor(
    np.column_stack([L_g.ravel(), C_g.ravel()]), dtype=torch.float32
)

# --- Prior surface (untrained model): uniform / flat mean ─────────────────────
model_prior = WorkingMemoryGP(
    inducing_grid_size       = mcfg["inducing_grid_size"],
    loc_lengthscale          = mcfg["loc_lengthscale"],
    color_lengthscale        = mcfg["color_lengthscale"],
    learn_inducing_locations = False,   # fixed for prior demo
).to(device)
likelihood_prior = gpytorch.likelihoods.GaussianLikelihood().to(device)

model_prior.eval(); likelihood_prior.eval()
with torch.no_grad():
    prior_mean = likelihood_prior(model_prior(grid_t)).mean.numpy().reshape(SURF_RES, SURF_RES)

ind_pts_prior = model_prior.variational_strategy.inducing_points.detach().cpu().numpy()  # (M**2, 2)
ind_vals_prior = np.zeros(len(ind_pts_prior))   # flat prior → z=0 for inducing points

# --- Posterior surface (trained model after encoding) ─────────────────────────
model_post = WorkingMemoryGP(
    inducing_grid_size       = mcfg["inducing_grid_size"],
    loc_lengthscale          = mcfg["loc_lengthscale"],
    color_lengthscale        = mcfg["color_lengthscale"],
    learn_inducing_locations = mcfg["learn_inducing_locations"],
).to(device)
likelihood_post = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
).to(device)
likelihood_post.noise = cfg["likelihood"]["noise_variance"]
likelihood_post.noise_covar.raw_noise.requires_grad_(False)

optimizer = torch.optim.Adam(model_post.parameters(), lr=cfg["training"]["encoding_lr"])
mll       = gpytorch.mlls.VariationalELBO(likelihood_post, model_post, num_data=len(samples))

model_post.train(); likelihood_post.train()
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

Z_MAX_POST = float(post_mean.max()) * 1.15   # z-scale for panels 3 & 4

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Shared axis-setup helper
# ══════════════════════════════════════════════════════════════════════════════

def _setup_ax(ax, z_max):
    ax.set_xlabel("Location (°)", labelpad=6, fontsize=10)
    ax.set_ylabel("Color (°)",    labelpad=6, fontsize=10)
    ax.set_zlabel("Weight / Activation", labelpad=6, fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_zlim(0, z_max)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.tick_params(axis='both', which='major', labelsize=7)


def _draw_items_at_base(ax, z_base=0.0):
    """Draw true-item star markers at z = z_base."""
    for d in per_item:
        ax.scatter(
            [d["true_loc"]], [d["true_col"]], [z_base],
            color=d["color"], marker="*", s=STAR_SIZE,
            edgecolors="white", linewidths=0.8, depthshade=False, zorder=6,
        )


def _draw_sample_cloud(ax):
    """Draw the noisy sample cloud coloured per item."""
    for d in per_item:
        ax.scatter(
            d["locs"], d["colors"], d["weights"],
            color=d["color"], alpha=ALPHA_CLOUD, s=MARKER_SIZE,
            depthshade=True,
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
_setup_ax(ax1, Z_MAX)
_draw_items_at_base(ax1, z_base=0.0)
ax1.set_title("Panel 1 — True Items", fontsize=11, pad=8)
_save(fig1, "panel1_true_items.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Panel 1 + sample cloud
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 2 …")
fig2, ax2 = _new_fig()
_setup_ax(ax2, Z_MAX)
_draw_items_at_base(ax2, z_base=0.0)
_draw_sample_cloud(ax2)
ax2.set_title("Panel 2 — True Items + Sample Cloud", fontsize=11, pad=8)
_save(fig2, "panel2_sample_cloud.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Panel 2 + inducing points on the PRIOR surface (flat)
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 3 …")
fig3, ax3 = _new_fig()
_setup_ax(ax3, Z_MAX)
_draw_items_at_base(ax3, z_base=0.0)
_draw_sample_cloud(ax3)

# Prior surface (semi-transparent)
ax3.plot_surface(
    L_g, C_g, prior_mean,
    cmap="viridis", edgecolor="none", alpha=0.35,vmin=0, vmax=Z_MAX,
)

# Inducing points on the prior surface
ax3.scatter(
    ind_pts_prior[:, 0], ind_pts_prior[:, 1], ind_vals_prior,
    color="white", edgecolors="gray", s=40, linewidths=0.5,
    depthshade=False, zorder=8, label="Inducing pts (prior)",
)
ax3.set_title("Panel 3 — + Inducing Points (Prior)", fontsize=11, pad=8)
ax3.legend(fontsize=7, loc="upper right")
_save(fig3, "panel3_prior.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Panel 2 + inducing points & surface AFTER encoding
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering Panel 4 …")
fig4, ax4 = _new_fig()
_setup_ax(ax4, Z_MAX_POST)
_draw_items_at_base(ax4, z_base=0.0)
_draw_sample_cloud(ax4)

# Posterior surface (trained)
ax4.plot_surface(
    L_g, C_g, post_mean,
    cmap="viridis", edgecolor="none", alpha=0.55,
)

# Inducing points on the posterior surface
ax4.scatter(
    ind_pts_post[:, 0], ind_pts_post[:, 1], ind_vals_post,
    color="red", edgecolors="white", s=50, linewidths=0.5,
    depthshade=False, zorder=8, label="Inducing pts (posterior)",
)
ax4.set_title("Panel 4 — + Inducing Points (Posterior)", fontsize=11, pad=8)
ax4.legend(fontsize=7, loc="upper right")
_save(fig4, "panel4_posterior.pdf")

print("\nDone!  All 4 PDFs saved to:", SAVE_DIR)
