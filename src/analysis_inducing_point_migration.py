"""
Analysis 4: Inducing Point Migration During Retrocue Maintenance

Tracks how inducing points physically migrate in (location, color) space
when a retrocue selectively weights the maintenance objective. This is
analogous to "representational resource reallocation" — the model's finite
representational substrate moves toward the attended item.

Produces:
  1. A trajectory plot showing inducing point positions pre- vs post-cue
  2. A quantitative panel showing mean distance from cued vs uncued items
  3. An animation of the migration over maintenance epochs
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gp_model import WorkingMemoryGP
from generator import generate_items, sample_training_data
from attention_mechanisms import SpatialProximityAttention
from simulation import load_config
import gpytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(_REPO_ROOT, "visualizations", "inducing_point_migration")
os.makedirs(SAVE_DIR, exist_ok=True)


def circular_distance(a, b):
    """Minimum circular distance on [-180, 180)."""
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def evaluate_items_at_positions(model, likelihood, items):
    """
    Evaluate the GP posterior mean at each item's (location, color) coordinate.
    Returns a vector of length N_items — the "activation" for each stored item.
    """
    model.eval()
    coords = torch.tensor(items, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = likelihood(model(coords))
        return pred.mean.cpu().numpy()


def run_trial_with_tracking(items, config, cued_item_idx=0):
    """
    Run a full trial (encode + maintain with retrocue) and return
    inducing point positions AND values at every epoch.
    """
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config['data']['n_samples_per_item'],
        loc_std=config['data']['loc_std'],
        color_std=config['data']['color_std'],
        loc_encoding_noise_std=config['data']['loc_encoding_noise_std'],
        color_encoding_noise_std=config['data']['color_encoding_noise_std']
    )

    model = WorkingMemoryGP(
        inducing_grid_size=config['model']['inducing_grid_size'],
        loc_lengthscale=config['model']['loc_lengthscale'],
        color_lengthscale=config['model']['color_lengthscale'],
        learn_inducing_locations=True,
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['encoding_lr'])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))

    # Track inducing points AND values
    ind_pts_history = []   # list of (M, 2) arrays — positions
    ind_vals_history = []  # list of (M,) arrays — variational means
    item_activations = []  # list of (N_items,) — posterior mean at item coords
    phase_labels = []      # 'encoding', 'pre_cue', 'post_cue'

    def snapshot(phase):
        pts = model.variational_strategy.inducing_points.detach().cpu().numpy().copy()
        vals = model.variational_strategy.variational_distribution.mean.detach().cpu().numpy().copy()
        acts = evaluate_items_at_positions(model, likelihood, items)
        ind_pts_history.append(pts)
        ind_vals_history.append(vals)
        item_activations.append(acts)
        phase_labels.append(phase)
        model.train()

    # ─── ENCODING ───
    model.train()
    likelihood.train()
    encoding_epochs = config['training']['encoding_epochs']
    for epoch in range(encoding_epochs):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        snapshot('encoding')

    # ─── MAINTENANCE ───
    maintenance_epochs = config['training']['maintenance_epochs']
    maint_lr = config['training']['maintenance_lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = maint_lr

    maint_grid_size = config['model'].get('maint_grid_size', 30)
    grid_1d = torch.linspace(-180.0, 180.0, maint_grid_size + 1, device=device)[:-1]
    grid_loc, grid_color = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    maint_grid = torch.stack([grid_loc.reshape(-1), grid_color.reshape(-1)], dim=1)

    with torch.no_grad():
        maint_weights = likelihood(model(maint_grid)).mean.detach()

    cued_loc = items[cued_item_idx][0]
    attn = SpatialProximityAttention(
        spatial_std=config['attention']['spatial_std'],
        attended_gain=config['attention']['attended_gain']
    ).to(device)
    cued_attn_weights = attn(maint_grid[:, 0], cued_loc)
    neutral_weights = torch.ones(len(maint_grid), device=device)

    cue_start_epoch = config['training']['cue_start_epoch']
    beta = config['training']['beta']

    for epoch in range(maintenance_epochs):
        optimizer.zero_grad()
        output = model(maint_grid)

        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)

        exp_ll = likelihood.expected_log_prob(maint_weights, output)

        if epoch >= cue_start_epoch:
            attn_weights = cued_attn_weights
            phase = 'post_cue'
        else:
            attn_weights = neutral_weights
            phase = 'pre_cue'

        weighted_ll = (exp_ll * attn_weights).sum() / len(maint_grid)
        loss = -weighted_ll + kl_div * beta
        loss.backward()
        optimizer.step()
        snapshot(phase)

    return (ind_pts_history, ind_vals_history, item_activations,
            phase_labels, items, cued_item_idx, cue_start_epoch)


def plot_migration_trajectories(ind_pts_history, phase_labels, items, cued_item_idx,
                                cue_start_epoch, encoding_epochs):
    """
    Static figure: inducing point positions at three key timepoints with
    displacement arrows from pre-cue to post-cue.
    """
    n_encoding = encoding_epochs
    pre_cue_end = n_encoding + cue_start_epoch - 1
    post_cue_end = len(ind_pts_history) - 1

    pts_post_encode = ind_pts_history[n_encoding - 1]
    pts_pre_cue = ind_pts_history[pre_cue_end]
    pts_post_cue = ind_pts_history[post_cue_end]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle("Inducing Point Migration During Retrocue Maintenance",
                 fontsize=14, fontweight='bold')

    cued_loc, cued_col = items[cued_item_idx]

    for ax, pts, title in zip(axes,
                              [pts_post_encode, pts_pre_cue, pts_post_cue],
                              ["After Encoding", "Pre-Cue (neutral maint.)", "Post-Cue (attended maint.)"]):
        ax.set_xlim(-185, 185)
        ax.set_ylim(-185, 185)
        ax.set_xlabel("Location (°)")
        ax.set_ylabel("Color (°)")
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.axhline(0, color='grey', lw=0.3)
        ax.axvline(0, color='grey', lw=0.3)

        # Plot inducing points
        ax.scatter(pts[:, 0], pts[:, 1], s=12, c='steelblue', alpha=0.6, zorder=2)

        # Plot items
        for i, (loc, col) in enumerate(items):
            marker = '*' if i == cued_item_idx else 'o'
            color = 'red' if i == cued_item_idx else 'dimgray'
            ax.scatter(loc, col, s=200, c=color, marker=marker, edgecolors='k',
                       linewidths=0.8, zorder=5,
                       label=f"Item {i} ({'cued' if i == cued_item_idx else 'uncued'})" if ax == axes[0] else "")

    # On the third panel, draw displacement arrows from pre_cue to post_cue
    ax = axes[2]
    displacements = pts_post_cue - pts_pre_cue
    mag = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    significant = mag > 1.0  # only show arrows > 1 degree displacement
    for i in np.where(significant)[0]:
        ax.annotate("", xy=pts_post_cue[i], xytext=pts_pre_cue[i],
                    arrowprops=dict(arrowstyle='->', color='orangered',
                                    lw=0.8, alpha=0.7))

    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.8)

    path = os.path.join(SAVE_DIR, "inducing_point_positions.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_distance_timecourse(ind_pts_history, phase_labels, items, cued_item_idx,
                             encoding_epochs):
    """
    Quantitative panel: mean spatial distance from inducing points to
    cued item vs. each uncued item, over all epochs.
    """
    cued_loc = items[cued_item_idx][0]
    uncued_locs = [items[i][0] for i in range(len(items)) if i != cued_item_idx]

    epochs = np.arange(len(ind_pts_history))
    mean_dist_cued = []
    mean_dist_uncued = []

    for pts in ind_pts_history:
        locs = pts[:, 0]
        d_cued = circular_distance(locs, cued_loc).mean()
        d_uncued = np.mean([circular_distance(locs, ul).mean() for ul in uncued_locs])
        mean_dist_cued.append(d_cued)
        mean_dist_uncued.append(d_uncued)

    mean_dist_cued = np.array(mean_dist_cued)
    mean_dist_uncued = np.array(mean_dist_uncued)

    # Find phase boundaries
    cue_onset = encoding_epochs + int(phase_labels[encoding_epochs:].index('post_cue') if 'post_cue' in phase_labels[encoding_epochs:] else 0)
    maint_start = encoding_epochs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Inducing Point Distance from Items Over Time", fontsize=13, fontweight='bold')

    # Panel 1: raw distances
    ax1.plot(epochs, mean_dist_cued, color='red', lw=1.8, label='Mean dist. to cued item (location)')
    ax1.plot(epochs, mean_dist_uncued, color='grey', lw=1.8, label='Mean dist. to uncued items (location)')
    ax1.axvline(maint_start, color='blue', ls='--', lw=1, alpha=0.7, label='Maintenance onset')
    ax1.axvline(cue_onset, color='green', ls='--', lw=1, alpha=0.7, label='Retrocue onset')
    ax1.set_ylabel("Mean circular distance (°)")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_title("Spatial (location) distance of inducing points from items")

    # Panel 2: difference (uncued - cued) → positive = points are closer to cued
    diff = mean_dist_uncued - mean_dist_cued
    ax2.plot(epochs, diff, color='purple', lw=1.8)
    ax2.axhline(0, color='grey', ls='-', lw=0.5)
    ax2.axvline(maint_start, color='blue', ls='--', lw=1, alpha=0.7)
    ax2.axvline(cue_onset, color='green', ls='--', lw=1, alpha=0.7)
    ax2.fill_between(epochs, 0, diff, where=diff > 0, alpha=0.2, color='green',
                     label='Closer to cued item')
    ax2.fill_between(epochs, 0, diff, where=diff < 0, alpha=0.2, color='red',
                     label='Closer to uncued items')
    ax2.set_ylabel("Dist(uncued) − Dist(cued) (°)")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Relative proximity bias (positive = points biased toward cued item)")
    ax2.legend(fontsize=9)

    path = os.path.join(SAVE_DIR, "distance_timecourse.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_migration_animation(ind_pts_history, phase_labels, items, cued_item_idx,
                             encoding_epochs, fps=15):
    """
    Animated GIF of inducing point positions over maintenance epochs.
    """
    maint_start = encoding_epochs
    # Subsample: every 2nd epoch during maintenance for smoother animation
    maint_indices = list(range(maint_start, len(ind_pts_history), 2))
    if maint_indices[-1] != len(ind_pts_history) - 1:
        maint_indices.append(len(ind_pts_history) - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-185, 185)
    ax.set_ylim(-185, 185)
    ax.set_xlabel("Location (°)", fontsize=11)
    ax.set_ylabel("Color (°)", fontsize=11)
    ax.set_aspect('equal')

    # Static item markers
    for i, (loc, col) in enumerate(items):
        marker = '*' if i == cued_item_idx else 'o'
        color = 'red' if i == cued_item_idx else 'dimgray'
        size = 250 if i == cued_item_idx else 150
        ax.scatter(loc, col, s=size, c=color, marker=marker, edgecolors='k',
                   linewidths=1, zorder=5)

    scat = ax.scatter([], [], s=14, c='steelblue', alpha=0.7, zorder=2)
    title_text = ax.set_title("", fontsize=11)

    # Reference positions at start of maintenance
    ref_pts = ind_pts_history[maint_start]

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat, title_text

    cue_start_maint = int(next(
        (i - maint_start for i, l in enumerate(phase_labels) if l == 'post_cue'),
        len(maint_indices)
    ))

    def update(frame_idx):
        idx = maint_indices[frame_idx]
        pts = ind_pts_history[idx]
        scat.set_offsets(pts)

        maint_epoch = idx - maint_start
        phase = phase_labels[idx]
        phase_str = "PRE-CUE (neutral)" if phase == 'pre_cue' else "POST-CUE (attended)"
        title_text.set_text(f"Maintenance epoch {maint_epoch}  |  {phase_str}")

        if phase == 'post_cue':
            scat.set_color('orangered')
            scat.set_alpha(0.6)
        else:
            scat.set_color('steelblue')
            scat.set_alpha(0.7)

        return scat, title_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(maint_indices), interval=1000//fps, blit=False)

    path = os.path.join(SAVE_DIR, "inducing_point_migration.gif")
    anim.save(path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_inducing_value_dynamics(ind_pts_history, ind_vals_history, phase_labels,
                                items, cued_item_idx, encoding_epochs):
    """
    The key analysis: track how inducing VALUES (analogous to neural firing
    rates) change during maintenance. Group inducing points by their proximity
    to each item and show selective preservation vs. decay.
    """
    cued_loc, cued_col = items[cued_item_idx]
    n_epochs = len(ind_vals_history)
    M = ind_vals_history[0].shape[0]

    # Assign each inducing point to its nearest item (by spatial location)
    # Use the post-encoding positions as reference
    ref_pts = ind_pts_history[encoding_epochs - 1]

    item_locs = np.array([it[0] for it in items])
    item_cols = np.array([it[1] for it in items])

    # For each inducing point, compute 2D circular distance to each item
    def circ_dist_2d(pt_loc, pt_col, it_loc, it_col):
        dl = circular_distance(pt_loc, it_loc)
        dc = circular_distance(pt_col, it_col)
        return np.sqrt(dl**2 + dc**2)

    # Assign each inducing point to nearest item
    assignments = np.zeros(M, dtype=int)
    for m in range(M):
        dists = [circ_dist_2d(ref_pts[m, 0], ref_pts[m, 1], items[i][0], items[i][1])
                 for i in range(len(items))]
        assignments[m] = np.argmin(dists)

    # Track mean absolute value for each item's cluster over time
    epochs = np.arange(n_epochs)
    cluster_means = {i: [] for i in range(len(items))}
    cluster_maxes = {i: [] for i in range(len(items))}

    for t in range(n_epochs):
        vals = ind_vals_history[t]
        for i in range(len(items)):
            mask = assignments == i
            if mask.sum() > 0:
                cluster_means[i].append(np.abs(vals[mask]).mean())
                cluster_maxes[i].append(np.abs(vals[mask]).max())
            else:
                cluster_means[i].append(0.0)
                cluster_maxes[i].append(0.0)

    # Also track total energy (L2 norm) per cluster
    cluster_energy = {i: [] for i in range(len(items))}
    for t in range(n_epochs):
        vals = ind_vals_history[t]
        for i in range(len(items)):
            mask = assignments == i
            cluster_energy[i].append(np.sqrt((vals[mask]**2).sum()) if mask.sum() > 0 else 0.0)

    maint_start = encoding_epochs
    cue_onset = maint_start + next(
        (i for i, l in enumerate(phase_labels[maint_start:]) if l == 'post_cue'), 0
    )

    # ─── Figure: 3 panels ───
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("Inducing Value Dynamics: Selective Preservation vs. Decay",
                 fontsize=14, fontweight='bold')

    # Panel 1: Mean |value| per item cluster
    ax = axes[0]
    for i in range(len(items)):
        lw = 2.5 if i == cued_item_idx else 1.2
        ls = '-' if i == cued_item_idx else '--'
        color = 'red' if i == cued_item_idx else f'C{i}'
        label = f"Item {i} ({'CUED' if i == cued_item_idx else 'uncued'})"
        ax.plot(epochs, cluster_means[i], lw=lw, ls=ls, color=color, label=label)
    ax.axvline(maint_start, color='blue', ls='--', lw=1, alpha=0.7, label='Maint. onset')
    ax.axvline(cue_onset, color='green', ls='--', lw=1, alpha=0.7, label='Retrocue onset')
    ax.set_ylabel("Mean |inducing value|")
    ax.set_xlabel("Epoch")
    ax.set_title("Mean activation magnitude per item cluster")
    ax.legend(fontsize=8, ncol=2, loc='upper right')

    # Panel 2: Energy (L2 norm) per cluster — "representational resource"
    ax = axes[1]
    for i in range(len(items)):
        lw = 2.5 if i == cued_item_idx else 1.2
        ls = '-' if i == cued_item_idx else '--'
        color = 'red' if i == cued_item_idx else f'C{i}'
        ax.plot(epochs, cluster_energy[i], lw=lw, ls=ls, color=color,
                label=f"Item {i} ({'CUED' if i == cued_item_idx else 'uncued'})")
    ax.axvline(maint_start, color='blue', ls='--', lw=1, alpha=0.7)
    ax.axvline(cue_onset, color='green', ls='--', lw=1, alpha=0.7)
    ax.set_ylabel("L2 norm (energy)")
    ax.set_xlabel("Epoch")
    ax.set_title("Representational energy per item cluster (analogous to population response magnitude)")
    ax.legend(fontsize=8, ncol=2, loc='upper right')

    # Panel 3: Ratio of cued / uncued energy — "attention gain"
    ax = axes[2]
    uncued_mean_energy = np.mean(
        [cluster_energy[i] for i in range(len(items)) if i != cued_item_idx], axis=0
    )
    cued_energy = np.array(cluster_energy[cued_item_idx])
    ratio = cued_energy / (uncued_mean_energy + 1e-8)
    ax.plot(epochs, ratio, lw=2, color='purple')
    ax.axhline(1.0, color='grey', ls='-', lw=0.5)
    ax.axvline(maint_start, color='blue', ls='--', lw=1, alpha=0.7, label='Maint. onset')
    ax.axvline(cue_onset, color='green', ls='--', lw=1, alpha=0.7, label='Retrocue onset')
    ax.set_ylabel("Energy ratio (cued / uncued)")
    ax.set_xlabel("Epoch")
    ax.set_title("Cued-to-uncued representational ratio (>1 = selective enhancement)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    path = os.path.join(SAVE_DIR, "inducing_value_dynamics.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    return assignments, cluster_energy


def plot_value_heatmap(ind_pts_history, ind_vals_history, phase_labels,
                       items, cued_item_idx, encoding_epochs):
    """
    Heatmap of inducing values over the (location, color) grid at three
    key timepoints, showing the cued item's bump preserved while others decay.
    """
    grid_size = int(np.sqrt(len(ind_vals_history[0])))
    maint_start = encoding_epochs
    cue_onset = maint_start + next(
        (i for i, l in enumerate(phase_labels[maint_start:]) if l == 'post_cue'), 0
    )

    timepoints = {
        'After Encoding': maint_start - 1,
        'Pre-Cue End': cue_onset - 1,
        'Post-Cue End': len(ind_vals_history) - 1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Inducing Values (\"Neural Activations\") Across Trial Phases",
                 fontsize=13, fontweight='bold')

    vmin = min(ind_vals_history[t].min() for t in timepoints.values())
    vmax = max(ind_vals_history[t].max() for t in timepoints.values())

    for ax, (title, t_idx) in zip(axes, timepoints.items()):
        pts = ind_pts_history[t_idx]
        vals = ind_vals_history[t_idx]

        sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=60, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.3,
                        zorder=2)

        for i, (loc, col) in enumerate(items):
            marker = '*' if i == cued_item_idx else 'o'
            color = 'lime' if i == cued_item_idx else 'white'
            ax.scatter(loc, col, s=250, c=color, marker=marker,
                       edgecolors='k', linewidths=1.5, zorder=5)

        ax.set_xlim(-185, 185)
        ax.set_ylim(-185, 185)
        ax.set_xlabel("Location (°)")
        ax.set_ylabel("Color (°)")
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')

    fig.colorbar(sc, ax=axes, shrink=0.8, label="Inducing value (variational mean)")

    path = os.path.join(SAVE_DIR, "inducing_value_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_posterior_at_items(item_activations, phase_labels, items, cued_item_idx,
                           encoding_epochs):
    """
    THE KEY PLOT: Posterior mean evaluated at each item's true (location, color)
    coordinate over time. This directly measures how well each memory is preserved.
    Analogous to "decodable information about each item" in neural data.
    """
    n_epochs = len(item_activations)
    n_items = len(items)
    epochs = np.arange(n_epochs)
    acts = np.array(item_activations)  # (n_epochs, n_items)

    maint_start = encoding_epochs
    cue_onset = maint_start + next(
        (i for i, l in enumerate(phase_labels[maint_start:]) if l == 'post_cue'), 0
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Representational Transformation:\nPosterior Activation at Each Item's True Coordinates",
                 fontsize=13, fontweight='bold')

    # Panel 1: Activation at each item over time
    ax = axes[0]
    for i in range(n_items):
        lw = 2.8 if i == cued_item_idx else 1.3
        ls = '-' if i == cued_item_idx else '--'
        color = 'red' if i == cued_item_idx else f'C{i}'
        label = (f"Item {i} @ ({items[i][0]:.0f}°, {items[i][1]:.0f}°) "
                 f"[{'CUED' if i == cued_item_idx else 'uncued'}]")
        ax.plot(epochs, acts[:, i], lw=lw, ls=ls, color=color, label=label)

    ax.axvline(maint_start, color='blue', ls='--', lw=1.2, alpha=0.7, label='Maintenance onset')
    ax.axvline(cue_onset, color='green', ls='--', lw=1.2, alpha=0.7, label='Retrocue onset')
    ax.axhline(0, color='grey', ls='-', lw=0.4)

    # Shade phases
    ax.axvspan(0, maint_start, alpha=0.04, color='blue', label='Encoding')
    ax.axvspan(maint_start, cue_onset, alpha=0.04, color='orange')
    ax.axvspan(cue_onset, n_epochs, alpha=0.06, color='green')

    ax.set_ylabel("Posterior mean f(loc_i, color_i)", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_title("Memory strength at each item's stored coordinates")
    ax.legend(fontsize=8, loc='upper left', ncol=2)

    # Panel 2: Cued item relative to mean of uncued items
    ax = axes[1]
    cued_act = acts[:, cued_item_idx]
    uncued_mean = np.mean(acts[:, [i for i in range(n_items) if i != cued_item_idx]], axis=1)

    ax.plot(epochs, cued_act, lw=2.5, color='red', label='Cued item activation')
    ax.plot(epochs, uncued_mean, lw=2, color='grey', ls='--', label='Mean uncued activation')
    ax.fill_between(epochs, uncued_mean, cued_act,
                    where=cued_act > uncued_mean, alpha=0.15, color='green',
                    label='Cued > Uncued')
    ax.fill_between(epochs, uncued_mean, cued_act,
                    where=cued_act <= uncued_mean, alpha=0.15, color='red')

    ax.axvline(maint_start, color='blue', ls='--', lw=1.2, alpha=0.7)
    ax.axvline(cue_onset, color='green', ls='--', lw=1.2, alpha=0.7)
    ax.axhline(0, color='grey', ls='-', lw=0.4)
    ax.set_ylabel("Posterior mean", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_title("Selective enhancement: cued vs. uncued item activation")
    ax.legend(fontsize=9, loc='upper left')

    path = os.path.join(SAVE_DIR, "posterior_at_items.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_color_profiles_transformation(ind_pts_history, ind_vals_history, phase_labels,
                                       items, cued_item_idx, encoding_epochs, config):
    """
    THE MOST DIRECT ANALOGY to subspace transformation:
    Show the color profile (GP posterior along color axis) at each item's
    spatial location, at three timepoints. This reveals:
    - Encoding: sharp peaks at correct colors for all items
    - Pre-cue: peaks maintained but perhaps slightly degraded
    - Post-cue: cued item's peak preserved/sharpened, uncued items flattened

    This is the WM-GP analogue of the orthogonal→parallel transformation:
    the "subspaces" here are the color profiles at different locations, and
    the "transformation" is selective sharpening vs. decay.
    """
    from gp_model import WorkingMemoryGP

    n_items = len(items)
    maint_start = encoding_epochs
    cue_onset = maint_start + next(
        (i for i, l in enumerate(phase_labels[maint_start:]) if l == 'post_cue'), 0
    )

    timepoints = {
        'After Encoding\n(all items encoded)': maint_start - 1,
        'Pre-Cue End\n(neutral maintenance)': cue_onset - 1,
        'Post-Cue End\n(attended maintenance)': len(ind_vals_history) - 1,
    }

    # For each timepoint, rebuild the model state and evaluate color profiles
    color_samples = np.linspace(-180, 180, 360)

    fig, axes = plt.subplots(n_items, 3, figsize=(14, 3 * n_items),
                             constrained_layout=True, sharey='row')
    fig.suptitle("Color Profile Transformation at Each Item's Location\n"
                 "(WM-GP analogue of subspace transformation upon retrocueing)",
                 fontsize=13, fontweight='bold')

    # We need to reconstruct model at each timepoint and query it
    # Instead, run the trial 3 times stopping at different points — but that's expensive.
    # Better: track full model states. Since we have inducing pts and vals,
    # we can reconstruct the prediction directly using the GP conditional.
    # However, for simplicity, let's re-run the trial once more with model snapshots.
    # Actually, the cleanest approach: run one more trial saving the model at key epochs.

    # Re-run the trial, saving model objects at 3 timepoints
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config['data']['n_samples_per_item'],
        loc_std=config['data']['loc_std'],
        color_std=config['data']['color_std'],
        loc_encoding_noise_std=config['data']['loc_encoding_noise_std'],
        color_encoding_noise_std=config['data']['color_encoding_noise_std']
    )
    torch.manual_seed(config['experiment']['random_seed'])
    np.random.seed(config['experiment']['random_seed'])

    model = WorkingMemoryGP(
        inducing_grid_size=config['model']['inducing_grid_size'],
        loc_lengthscale=config['model']['loc_lengthscale'],
        color_lengthscale=config['model']['color_lengthscale'],
        learn_inducing_locations=True,
    ).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['encoding_lr'])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))

    def get_color_profiles():
        model.eval()
        likelihood.eval()
        profiles = {}
        for i, (loc, col) in enumerate(items):
            query = torch.tensor(
                [[loc, c] for c in color_samples], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                pred = likelihood(model(query))
                profiles[i] = pred.mean.cpu().numpy()
        model.train()
        likelihood.train()
        return profiles

    saved_profiles = {}
    target_epochs = list(timepoints.values())

    # Encoding
    model.train()
    likelihood.train()
    enc_epochs = config['training']['encoding_epochs']
    for epoch in range(enc_epochs):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        if epoch == target_epochs[0]:
            saved_profiles['After Encoding\n(all items encoded)'] = get_color_profiles()

    if enc_epochs - 1 == target_epochs[0]:
        saved_profiles['After Encoding\n(all items encoded)'] = get_color_profiles()

    # Maintenance
    maint_lr = config['training']['maintenance_lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = maint_lr
    maint_grid_size = config['model'].get('maint_grid_size', 30)
    grid_1d = torch.linspace(-180.0, 180.0, maint_grid_size + 1, device=device)[:-1]
    grid_loc, grid_color = torch.meshgrid(grid_1d, grid_1d, indexing='ij')
    maint_grid = torch.stack([grid_loc.reshape(-1), grid_color.reshape(-1)], dim=1)
    with torch.no_grad():
        model.eval()
        maint_weights_t = likelihood(model(maint_grid)).mean.detach()
        model.train()

    cued_loc_val = items[cued_item_idx][0]
    attn_mod = SpatialProximityAttention(
        spatial_std=config['attention']['spatial_std'],
        attended_gain=config['attention']['attended_gain']
    ).to(device)
    cued_w = attn_mod(maint_grid[:, 0], cued_loc_val)
    neutral_w = torch.ones(len(maint_grid), device=device)
    cue_start = config['training']['cue_start_epoch']
    beta = config['training']['beta']
    maint_epochs = config['training']['maintenance_epochs']

    for epoch in range(maint_epochs):
        optimizer.zero_grad()
        output = model(maint_grid)
        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
        exp_ll = likelihood.expected_log_prob(maint_weights_t, output)
        aw = cued_w if epoch >= cue_start else neutral_w
        weighted_ll = (exp_ll * aw).sum() / len(maint_grid)
        loss = -weighted_ll + kl_div * beta
        loss.backward()
        optimizer.step()

        global_epoch = enc_epochs + epoch
        if global_epoch == target_epochs[1]:
            saved_profiles['Pre-Cue End\n(neutral maintenance)'] = get_color_profiles()
        if global_epoch == target_epochs[2]:
            saved_profiles['Post-Cue End\n(attended maintenance)'] = get_color_profiles()

    # If last epoch, capture it
    if 'Post-Cue End\n(attended maintenance)' not in saved_profiles:
        saved_profiles['Post-Cue End\n(attended maintenance)'] = get_color_profiles()

    # Now plot
    for row_idx in range(n_items):
        is_cued = (row_idx == cued_item_idx)
        item_loc, item_col = items[row_idx]

        for col_idx, (title, _) in enumerate(timepoints.items()):
            ax = axes[row_idx, col_idx] if n_items > 1 else axes[col_idx]
            profile = saved_profiles.get(title, {}).get(row_idx, np.zeros_like(color_samples))

            color = 'red' if is_cued else 'steelblue'
            ax.plot(color_samples, profile, lw=2, color=color)
            ax.axvline(item_col, color='black', ls=':', lw=1.5, alpha=0.7)
            ax.fill_between(color_samples, 0, profile, alpha=0.15, color=color)

            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                label = "CUED" if is_cued else "uncued"
                ax.set_ylabel(f"Item {row_idx} [{label}]\nloc={item_loc:.0f}°",
                              fontsize=9)
            if row_idx == n_items - 1:
                ax.set_xlabel("Color (°)")

            ax.set_xlim(-180, 180)

    path = os.path.join(SAVE_DIR, "color_profile_transformation.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_density_shift(ind_pts_history, phase_labels, items, cued_item_idx,
                       encoding_epochs):
    """
    Histogram of inducing point locations (1D, projected onto spatial axis)
    at three key timepoints, showing the density shift toward the cued location.
    """
    n_enc = encoding_epochs
    cue_start_idx = n_enc + next(
        (i for i, l in enumerate(phase_labels[n_enc:]) if l == 'post_cue'),
        0
    )
    post_cue_end = len(ind_pts_history) - 1

    pts_post_encode = ind_pts_history[n_enc - 1][:, 0]
    pts_pre_cue = ind_pts_history[cue_start_idx - 1][:, 0] if cue_start_idx > n_enc else pts_post_encode
    pts_post_cue = ind_pts_history[post_cue_end][:, 0]

    cued_loc = items[cued_item_idx][0]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    bins = np.linspace(-180, 180, 37)

    ax.hist(pts_post_encode, bins=bins, alpha=0.35, color='blue',
            label='After encoding', density=True)
    ax.hist(pts_pre_cue, bins=bins, alpha=0.35, color='orange',
            label='Pre-cue end', density=True)
    ax.hist(pts_post_cue, bins=bins, alpha=0.55, color='red',
            label='Post-cue end', density=True)

    for i, (loc, col) in enumerate(items):
        style = {'color': 'red', 'lw': 2.5, 'ls': '-'} if i == cued_item_idx else {'color': 'grey', 'lw': 1.5, 'ls': '--'}
        ax.axvline(loc, **style, alpha=0.8,
                   label=f"Item {i} loc ({'cued' if i == cued_item_idx else 'uncued'})")

    ax.set_xlabel("Location (°)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Spatial Distribution of Inducing Points: Density Shift Toward Cued Item",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    path = os.path.join(SAVE_DIR, "density_shift_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Analysis 4: Inducing Point Migration During Retrocue")
    print("=" * 60)

    config = load_config(filename="config_retrocue.yaml")
    n_items = 4
    seed = config['experiment']['random_seed']
    items = generate_items(n_items, seed=seed)
    cued_item_idx = 0

    print(f"\nItems (N={n_items}): {items}")
    print(f"Cued item: {cued_item_idx} → location={items[cued_item_idx][0]:.1f}°, "
          f"color={items[cued_item_idx][1]:.1f}°")
    print(f"Inducing grid: {config['model']['inducing_grid_size']}×{config['model']['inducing_grid_size']} "
          f"= {config['model']['inducing_grid_size']**2} points")
    print(f"Encoding epochs: {config['training']['encoding_epochs']}")
    print(f"Maintenance epochs: {config['training']['maintenance_epochs']}")
    print(f"Cue onset (within maintenance): epoch {config['training']['cue_start_epoch']}")
    print()

    print("Running trial with inducing point tracking...")
    (ind_pts_history, ind_vals_history, item_activations,
     phase_labels, items, cued_item_idx, cue_start_epoch) = \
        run_trial_with_tracking(items, config, cued_item_idx=cued_item_idx)
    encoding_epochs = config['training']['encoding_epochs']
    print(f"  Tracked {len(ind_pts_history)} snapshots "
          f"({encoding_epochs} encoding + {len(ind_pts_history) - encoding_epochs} maintenance)")

    # Compute summary statistics — positions
    pts_start = ind_pts_history[encoding_epochs]
    pts_end = ind_pts_history[-1]
    disp = np.sqrt(((pts_end - pts_start)**2).sum(axis=1))
    print(f"\n  Inducing point POSITION displacement (maintenance start → end):")
    print(f"    Mean: {disp.mean():.2f}°  |  Max: {disp.max():.2f}°  |  "
          f"Median: {np.median(disp):.2f}°")

    cued_loc = items[cued_item_idx][0]
    dist_start = circular_distance(pts_start[:, 0], cued_loc)
    dist_end = circular_distance(pts_end[:, 0], cued_loc)
    print(f"    Mean dist to cued loc: {dist_start.mean():.1f}° → {dist_end.mean():.1f}° "
          f"(Δ = {dist_end.mean() - dist_start.mean():.1f}°)")

    # Compute summary statistics — values
    vals_start = ind_vals_history[encoding_epochs]
    vals_end = ind_vals_history[-1]
    print(f"\n  Inducing VALUE dynamics (maintenance start → end):")
    print(f"    Total energy (L2): {np.linalg.norm(vals_start):.2f} → {np.linalg.norm(vals_end):.2f}")
    print(f"    Max value: {vals_start.max():.3f} → {vals_end.max():.3f}")
    print(f"    Mean |value|: {np.abs(vals_start).mean():.4f} → {np.abs(vals_end).mean():.4f}")

    print("\nGenerating visualizations...")

    # 1. Position-based analyses
    plot_migration_trajectories(ind_pts_history, phase_labels, items, cued_item_idx,
                                cue_start_epoch, encoding_epochs)
    plot_distance_timecourse(ind_pts_history, phase_labels, items, cued_item_idx,
                            encoding_epochs)
    plot_density_shift(ind_pts_history, phase_labels, items, cued_item_idx,
                       encoding_epochs)

    # 2. Value-based analyses (the main story)
    plot_posterior_at_items(item_activations, phase_labels, items, cued_item_idx,
                           encoding_epochs)
    plot_color_profiles_transformation(ind_pts_history, ind_vals_history, phase_labels,
                                       items, cued_item_idx, encoding_epochs, config)
    plot_inducing_value_dynamics(ind_pts_history, ind_vals_history, phase_labels,
                                items, cued_item_idx, encoding_epochs)
    plot_value_heatmap(ind_pts_history, ind_vals_history, phase_labels,
                       items, cued_item_idx, encoding_epochs)

    # 3. Animation
    plot_migration_animation(ind_pts_history, phase_labels, items, cued_item_idx,
                            encoding_epochs)

    print(f"\nAll outputs saved to: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
