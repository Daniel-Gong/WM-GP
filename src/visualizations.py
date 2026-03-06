import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gpytorch
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_COLORWHEEL: Optional[np.ndarray] = None  # cached (360, 3) float32 in [0, 1]

def _load_colorwheel() -> np.ndarray:
    """Return the 360×3 colorwheel array (RGB in [0,1]), loaded once from colorwheel.csv."""
    global _COLORWHEEL
    if _COLORWHEEL is not None:
        return _COLORWHEEL
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'colorwheel.csv'))
    df = pd.read_csv(csv_path, index_col=0)          # columns: R, G, B  (0-255 ints)
    arr = df[['R', 'G', 'B']].values.astype(np.float32) / 255.0  # (360, 3)
    _COLORWHEEL = arr
    return _COLORWHEEL

def _item_colors_from_wheel(color_degrees) -> np.ndarray:
    """Map item color values in [-180, 179] to colorwheel RGB. Returns (n, 3) array."""
    wheel = _load_colorwheel()
    indices = [int(c) + 180 for c in color_degrees]   # [-180,179] → [0,359]
    indices = [max(0, min(359, idx)) for idx in indices]
    return np.array([wheel[idx] for idx in indices])


def plot_gp_surface_2d(model, likelihood, items, epoch, prefix="", save_dir="visualizations", filename="gp_surface_final.png"):
    """
    Plots a 2D heatmap of the GP predictive Mean with Inducing Points overlaid.
    X-axis: Location (-pi, pi)
    Y-axis: Color (-pi, pi)
    """
    print("Plotting 2D GP Surface...")
    model.eval()
    likelihood.eval()
    
    res = 100
    locs = torch.linspace(-180.0, 180.0, res, device=device)
    colors = torch.linspace(-180.0, 180.0, res, device=device)
    
    L, C = torch.meshgrid(locs, colors, indexing='ij')
    grid = torch.stack([L.flatten(), C.flatten()], dim=-1)
    
    with torch.no_grad():
        preds = likelihood(model(grid))
        mean_surface = preds.mean.view(res, res).cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot predicted surface heatmap
    cax = ax.imshow(
        mean_surface.T, 
        extent=[-180.0, 180.0, -180.0, 180.0], 
        origin='lower', 
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(cax, label='Predictive Mean (Memory Strength)')
    
    # Plot true items – colour each star with its actual colorwheel colour
    true_locs = [i[0] for i in items]
    true_cols = [i[1] for i in items]
    item_rgb = _item_colors_from_wheel(true_cols)
    ax.scatter(true_locs, true_cols, c=item_rgb, marker='*', s=200,
               label='Encoded Items', edgecolors='white', linewidths=0.8)
    
    # Plot inducing points
    if hasattr(model, 'variational_strategy'):
        ind_pts = model.variational_strategy.inducing_points.detach().cpu().numpy()
        ax.scatter(ind_pts[:, 0], ind_pts[:, 1], c='white', marker='.', s=50, alpha=0.5, label='Inducing Pts')
        
    ax.set_title(f"{prefix} GP Surface - Epoch {epoch}")
    ax.set_xlabel("Location (deg)")
    ax.set_ylabel("Color (deg)")
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-180.0, 180.0])
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def plot_training_trajectories(history: Dict, save_dir="visualizations",filename="training_trajectories.png"):
    """
    Plots the trajectories of GP Hyperparameters (Lengthscales, Noise) 
    and ELBO Losses over the training epochs.
    """
    print("Plotting Training Trajectories...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Encoding Loss
    axs[0].plot(history.get('encoding_loss', []), color='blue', label='Encoding (-ELBO)')
    axs[0].set_title("Variational ELBO (Encoding)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Plot 2: Maintenance Loss (if any)
    if 'maintenance_loss' in history and len(history['maintenance_loss']) > 0:
        axs[1].plot(history['maintenance_loss'], color='orange', label='Maint (KL + Attn)')
        axs[1].set_title("Maintenance Loss")
    else:
        axs[1].text(0.5, 0.5, 'No Maintenance Run', ha='center')
    axs[1].set_xlabel("Epoch")
    
    # Plot 3: Specific Model Parameter Trajectories
    epochs = len(history.get('loc_lengthscale', []))
    if epochs > 0:
        axs[2].plot(history['loc_lengthscale'], label='Location LS', color='red')
        axs[2].plot(history['color_lengthscale'], label='Color LS', color='green')
        axs[2].set_title("Periodic Lengthscales")
        axs[2].set_xlabel("Epoch")
        axs[2].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

def plot_item_retrieval_errors(
    history: Dict,
    items,
    cued_item_idx: Optional[int] = None,
    cue_start_epoch: Optional[int] = None,
    save_dir: str = "visualizations",
    filename: str = "item_losses.png",
) -> None:
    """
    Two-panel figure showing per-item retrieval error over training epochs.

    Panel 1 – Encoding phase:  unsigned retrieval error per item.
    Panel 2 – Maintenance phase: unsigned retrieval error per item.
                If cued_item_idx is not None, a vertical dashed line marks
                cue_start_epoch in the maintenance panel.

    Parameters
    ----------
    history : dict
        Must contain 'unsigned_errors' (dict: item_idx -> list of floats),
        'encoding_loss', and optionally 'maintenance_loss'.
    items : list of (loc, color) tuples
        Used to derive colorwheel colours for each item's line.
    cued_item_idx : int or None
        Index of the cued item (highlighted with thicker line).
    cue_start_epoch : int or None
        Epoch within maintenance at which the retrocue is applied.
    save_dir : str
        Directory to save the figure.
    filename : str
        Output filename.
    """
    n_enc  = len(history.get('encoding_loss', []))
    n_maint = len(history.get('maintenance_loss', []))
    n_total = n_enc + n_maint
    errors  = history.get('unsigned_errors', {})

    if not errors:
        return   # nothing logged

    item_colors_rgb = _item_colors_from_wheel([it[1] for it in items])

    fig, (ax_enc, ax_maint) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for i, errs in errors.items():
        errs = np.array(errs)
        rgb  = item_colors_rgb[i]
        lw   = 2.5 if i == cued_item_idx else 1.5
        ls   = '-'  if i == cued_item_idx else '--'
        label = f"Item {i+1}" + (" ★ cued" if i == cued_item_idx else "")

        enc_errs   = errs[:n_enc]   if n_enc   > 0 else np.array([])
        maint_errs = errs[n_enc:]   if n_maint > 0 else np.array([])

        if len(enc_errs):
            ax_enc.plot(enc_errs, color=rgb, linewidth=lw, linestyle=ls, label=label)
        if len(maint_errs):
            ax_maint.plot(maint_errs, color=rgb, linewidth=lw, linestyle=ls, label=label)

    # ── Encoding panel ──────────────────────────────────────────────────────
    ax_enc.set_title("Encoding phase — retrieval error per item", fontsize=12)
    ax_enc.set_xlabel("Epoch", fontsize=10)
    ax_enc.set_ylabel("Unsigned error (°)", fontsize=10)
    ax_enc.grid(True, linestyle='--', alpha=0.4)
    ax_enc.legend(fontsize=9)

    # ── Maintenance panel ───────────────────────────────────────────────────
    ax_maint.set_title("Maintenance phase — retrieval error per item", fontsize=12)
    ax_maint.set_xlabel("Epoch", fontsize=10)
    ax_maint.set_ylabel("Unsigned error (°)", fontsize=10)
    ax_maint.grid(True, linestyle='--', alpha=0.4)
    ax_maint.legend(fontsize=9)

    if cued_item_idx is not None and cue_start_epoch is not None and n_maint > 0:
        ax_maint.axvline(
            x=cue_start_epoch,
            color='white', linestyle=':', linewidth=2.0,
            label=f"Cue onset (epoch {cue_start_epoch})",
        )
        ax_maint.text(
            cue_start_epoch + 0.5, ax_maint.get_ylim()[1] * 0.97,
            "cue onset", fontsize=8, va='top', color='grey',
        )
        ax_maint.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

def plot_signed_error_histogram(signed_errors: List[float], set_size: int, prefix="Neutral", save_dir="visualizations"):
    """
    Plots a histogram of the signed retrieval errors spanning [-pi, pi).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Bins between -180.0 and 180.0
    bins = np.linspace(-180.0, 180.0, 31)
    ax.hist(signed_errors, bins=bins, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_title(f"Signed Error Distribution (N={set_size}, {prefix})")
    ax.set_xlabel("Error (degrees)")
    ax.set_ylabel("Density")
    ax.set_xlim([-180.0, 180.0])
    
    # Set x-ticks to nicer labels
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['-180', '-90', '0', '90', '180'])
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"signed_error_hist_N{set_size}_{prefix}.png"), dpi=150)
    plt.close()

def create_gp_surface_2d_gif(history_surfaces: List[np.ndarray], history_ind_pts: List[np.ndarray], items: List[Tuple], save_dir="visualizations", filename="gp_optimization.gif"):
    """
    Animates the optimization of the GP Surface along with inducing points over epochs.
    Requires history_surfaces to contain eval-grid (50x50) mean outputs.
    """
    print("Creating 2D GP Optimization GIF...")
    if not history_surfaces:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cax = ax.imshow(
        history_surfaces[0].T, 
        extent=[-180.0, 180.0, -180.0, 180.0], 
        origin='lower', 
        cmap='viridis',
        aspect='auto',
        vmin=0.0,
        vmax=max(1.0, np.max([np.max(s) for s in history_surfaces]))
    )
    plt.colorbar(cax, label='Predictive Mean')
    
    # Plot true items (static)
    true_locs = [i[0] for i in items]
    true_cols = [i[1] for i in items]
    item_rgb = _item_colors_from_wheel([i[1] for i in items])
    ax.scatter(true_locs, true_cols, c=item_rgb, marker='*', s=200, label='Items', edgecolors='white', zorder=5)
    
    # Inducing points (dynamic)
    scatter_ind = ax.scatter(history_ind_pts[0][:, 0], history_ind_pts[0][:, 1], c='white', marker='.', s=60, alpha=0.8, label='Ind. Pts', zorder=4)
    
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-180.0, 180.0])
    ax.set_xlabel("Location (deg)")
    ax.set_ylabel("Color (deg)")
    ax.legend(loc='upper right')
    
    def update(frame):
        ax.set_title(f"GP Optimization - Epoch {frame}")
        cax.set_array(history_surfaces[frame].T)
        scatter_ind.set_offsets(history_ind_pts[frame])
        return cax, scatter_ind
        
    anim = FuncAnimation(fig, update, frames=len(history_surfaces), interval=100, blit=False)
    os.makedirs(save_dir, exist_ok=True)
    anim.save(os.path.join(save_dir, filename), dpi=100, writer='pillow')
    plt.close()

def create_gp_surface_3d_gif(history_surfaces: List[np.ndarray], history_ind_pts: List[np.ndarray], history_ind_vals: List[np.ndarray], items: List[Tuple], save_dir="visualizations", filename="gp_optimization_3d.gif", rotation_speed: float = 2.0):
    """
    Animates a 3D surface plot of the GP along with floating inducing points over epochs.
    Requires history_surfaces to contain eval-grid (50x50) mean outputs.
    """
    print("Creating 3D GP Optimization GIF...")
    if not history_surfaces:
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 100x100 grid that matches the eval grid in simulation.py
    res = 100
    locs = np.linspace(-180.0, 180.0, res)
    colors = np.linspace(-180.0, 180.0, res)
    L, C = np.meshgrid(locs, colors, indexing='ij')
    
    # Determine global Z limits
    z_max = max(1.0, np.max([np.max(s) for s in history_surfaces]))
    z_min = np.min([np.min(s) for s in history_surfaces])
    
    # Precompute item colours once (static across frames)
    item_rgb = _item_colors_from_wheel([i[1] for i in items])
    true_locs = [i[0] for i in items]
    true_cols = [i[1] for i in items]

    def update(frame):
        ax.clear()
        ax.set_title(f"3D GP Optimization - Epoch {frame}")
        ax.set_xlabel("Location (deg)")
        ax.set_ylabel("Color (deg)")
        ax.set_zlabel("Predictive Mean")
        ax.set_xlim([-180.0, 180.0])
        ax.set_ylim([-180.0, 180.0])
        ax.set_zlim([z_min, z_max])

        # Slowly rotate the camera
        azim = 45 + frame * rotation_speed
        ax.view_init(elev=25, azim=azim)
        
        # Plot surface
        surf = ax.plot_surface(L, C, history_surfaces[frame], cmap='viridis', edgecolor='none', alpha=0.7)
        
        # Plot inducing points in 3D
        ind_pts = history_ind_pts[frame]
        ind_vals = history_ind_vals[frame]
        ax.scatter(ind_pts[:, 0], ind_pts[:, 1], ind_vals, color='red', s=30, label='Ind. Pts', depthshade=False)
        
        # Plot items at the base plane for reference
        ax.scatter(true_locs, true_cols, [z_min]*len(items), c=item_rgb, marker='*', s=200, label='Items (Ground)', edgecolors='white')
        
        ax.legend(loc='upper right')
        return fig,

    anim = FuncAnimation(fig, update, frames=len(history_surfaces), interval=100, blit=False)
    
    os.makedirs(save_dir, exist_ok=True)
    anim.save(os.path.join(save_dir, filename), dpi=100, writer='pillow')

    # Save the last frame as a standalone PNG
    update(len(history_surfaces) - 1)
    last_frame_filename = os.path.splitext(filename)[0] + "_last_frame.png"
    fig.savefig(os.path.join(save_dir, last_frame_filename), dpi=150, bbox_inches='tight')
    print(f"  Saved last frame → {os.path.join(save_dir, last_frame_filename)}")
    plt.close()

def plot_set_size_effect(df: pd.DataFrame, save_dir="visualizations"):
    """
    Plots the mean absolute error vs Set Size with ±1 SD error bars.
    Error bars use the 'SD Signed Error' column when available.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    yerr = df['SD Signed Error'] if 'SD Signed Error' in df.columns else None

    ax.errorbar(
        df['Set Size'], df['Mean Abs Error'],
        yerr=yerr,
        marker='o', linestyle='-', color='b', markersize=8,
        capsize=5, capthick=1.5, elinewidth=1.5, ecolor='steelblue',
        label='Mean ± 1 SD'
    )
    ax.set_title("Set Size Effect on Retrieval Error", fontsize=14)
    ax.set_xlabel("Set Size (N)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (deg)", fontsize=12)

    ax.set_xticks(df['Set Size'])
    ax.grid(True, linestyle='--', alpha=0.7)
    if yerr is not None:
        ax.legend(fontsize=10)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "set_size_effect.png"), dpi=150)
    plt.close()

def plot_error_distributions(errors_per_set_size: dict, save_dir="visualizations"):
    """
    Plots a signed error distribution histogram for each set size.
    Each set size gets its own subplot panel (and individual PNG file).
    SD of the signed error distribution is annotated on each plot.

    Parameters
    ----------
    errors_per_set_size : dict
        Keys are set sizes (int), values are lists of signed errors (float, degrees).
    save_dir : str
        Directory to save the figures.
    """

    set_sizes = sorted(errors_per_set_size.keys())
    n = len(set_sizes)

    # ── 1. Individual histograms per set size (with KDE density curve) ───────
    for ss in set_sizes:
        errs = np.array(errors_per_set_size[ss])
        sd   = np.std(errs)

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-180.0, 180.0, 37)          # 10-deg bins
        ax.hist(errs, bins=bins, color='steelblue', edgecolor='black', alpha=0.75, density=True)

        # KDE density curve
        kde = stats.gaussian_kde(errs)
        x_kde = np.linspace(-180.0, 180.0, 300)
        ax.plot(x_kde, kde(x_kde), color='steelblue', linewidth=2.0, label='KDE')

        ax.axvline(0,  color='red',   linestyle='--', linewidth=1.8, label='0°')
        ax.axvline( sd, color='gray', linestyle=':',  linewidth=1.4, label=f'+SD ({sd:.1f}°)')
        ax.axvline(-sd, color='gray', linestyle=':',  linewidth=1.4, label=f'−SD ({sd:.1f}°)')

        ax.set_title(f"Signed Error Distribution  |  N = {ss}", fontsize=13)
        ax.set_xlabel("Signed Error (deg)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim([-180.0, 180.0])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.legend(fontsize=9, loc='upper right')

        # Annotate SD in the corner
        ax.text(0.02, 0.96, f"SD = {sd:.2f}°", transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"error_dist_N{ss}.png"), dpi=150)
        plt.close()
        print(f"  Saved error distribution for N={ss}  (SD={sd:.2f}°)")

    # ── 2. Combined overlay: all set sizes in one figure (different colours) ──
    palette = plt.cm.tab10.colors          # up to 10 distinct colours
    bins_ov  = np.linspace(-180.0, 180.0, 37)
    x_kde    = np.linspace(-180.0, 180.0, 300)

    fig_ov, ax_ov = plt.subplots(figsize=(8, 5))
    for k, ss in enumerate(set_sizes):
        errs  = np.array(errors_per_set_size[ss])
        color = palette[k % len(palette)]
        ax_ov.hist(errs, bins=bins_ov, alpha=0.35, density=True,
                   color=color, edgecolor='none')
        kde   = stats.gaussian_kde(errs)
        ax_ov.plot(x_kde, kde(x_kde), color=color, linewidth=2.2,
                   label=f'N={ss} KDE')

    ax_ov.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax_ov.set_title("Signed Error Distributions — All Set Sizes", fontsize=13)
    ax_ov.set_xlabel("Signed Error (deg)", fontsize=11)
    ax_ov.set_ylabel("Density", fontsize=11)
    ax_ov.set_xlim([-180.0, 180.0])
    ax_ov.set_xticks([-180, -90, 0, 90, 180])
    ax_ov.legend(fontsize=9, loc='upper right')
    ax_ov.grid(True, linestyle='--', alpha=0.5)
    fig_ov.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "error_distributions_overlay.png"), dpi=150)
    plt.close()
    print("  Saved combined overlay error distribution.")

    # ── 3. Combined panel (all set sizes side by side) ───────────────────────
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    bins = np.linspace(-180.0, 180.0, 37)
    for idx, ss in enumerate(set_sizes):
        row, col = divmod(idx, ncols)  # ncols=2 → 2x2 grid
        ax  = axes[row][col]
        errs = np.array(errors_per_set_size[ss])
        sd   = np.std(errs)

        ax.hist(errs, bins=bins, color='steelblue', edgecolor='black', alpha=0.75, density=True)
        ax.axvline(0,   color='red',  linestyle='--', linewidth=1.5)
        ax.axvline( sd, color='gray', linestyle=':',  linewidth=1.2)
        ax.axvline(-sd, color='gray', linestyle=':',  linewidth=1.2)

        ax.set_title(f"N = {ss}", fontsize=12)
        ax.set_xlabel("Signed Error (deg)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim([-180.0, 180.0])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.text(0.03, 0.95, f"SD = {sd:.2f}°", transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Signed Error Distributions by Set Size", fontsize=14, y=1.02)
    fig.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "error_distributions_combined.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved combined error distribution panel.")

def plot_retrocue_benefit(neutral_errors: List[float], cued_errors: List[float], set_size: int, save_dir="visualizations", p_val: Optional[float] = None):
    """
    Plots a bar chart comparing Neutral vs Cued Mean Absolute Errors with SEM error bars.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    conditions = ['Neutral', 'Cued']
    maes = [np.mean(neutral_errors), np.mean(cued_errors)]
    sems = [stats.sem(neutral_errors), stats.sem(cued_errors)]
    colors = ['lightcoral', 'lightgreen']
    
    ax.bar(conditions, maes, yerr=sems, color=colors, edgecolor='black', width=0.5, capsize=5)
    ax.set_title(f"Retrocue Benefit (N={set_size})", fontsize=14)
    ax.set_ylabel("Mean Absolute Error (deg)", fontsize=12)
    
    # Annotate bars
    for i, v in enumerate(maes):
        ax.text(i, v + sems[i] + max(0.5, v * 0.05), f"{v:.2f}", ha='center', fontweight='bold')
        
    if p_val is not None:
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
            
        y_max = max(maes[0] + sems[0], maes[1] + sems[1])
        h = max(1.0, 0.05 * y_max)
        ax.plot([0, 0, 1, 1], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.5, c='k')
        ax.text(0.5, y_max + 2.5*h, sig, ha='center', va='bottom', color='k', fontsize=12, fontweight='bold')
        
        # Adjust y-limit to fit annotations
        ax.set_ylim(0, y_max + 5*h)
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"retrocue_benefit_N={set_size}.png"), dpi=150)
    plt.close()

def plot_bias_effect(df: pd.DataFrame, save_dir="visualizations"):
    """
    Plots the Bias curve across color feature distances.
    Assumes Bias > 0 is Attraction (pulled towards distractor offset), Bias < 0 is Repulsion.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    dist_deg = df['Distance_deg']
    bias_deg = df['Bias_deg']
    
    yerr = df['SEM_deg'] if 'SEM_deg' in df.columns else None

    if yerr is not None:
        ax.errorbar(
            dist_deg, bias_deg, yerr=yerr,
            marker='o', linestyle='-', color='purple', markersize=8,
            capsize=5, capthick=1.5, elinewidth=1.5, ecolor='mediumpurple',
            label='Mean ± 1 SEM'
        )
        ax.legend(fontsize=10)
    else:
        ax.plot(dist_deg, bias_deg, marker='o', linestyle='-', color='purple', markersize=8)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    ax.set_title("Attraction/Repulsion Bias vs Featural Distance", fontsize=14)
    ax.set_xlabel("Target-Distractor Distance (degrees)", fontsize=12)
    ax.set_ylabel("Bias Magnitude (degrees)\n(-) Repulsion, (+) Attraction", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, linestyle='--', alpha=0.7)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "bias_effect.png"), dpi=150)
    plt.close()

def visualize_continuous_weighting(cued_location, maintenance_samples, proximity_labels):
    """
    Visualize how the continuous weighting works based on distance to cued location
    """
    import matplotlib.pyplot as plt
    
    # Compute distances for visualization
    distances_to_cued = torch.norm(maintenance_samples[:, :2] - cued_location, dim=1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Distance vs Weight
    ax1.scatter(distances_to_cued.cpu().numpy(), proximity_labels.cpu().numpy(), alpha=0.6)
    ax1.set_xlabel('Distance to Cued Location')
    ax1.set_ylabel('Continuous Weight')
    ax1.set_title('Distance vs Continuous Weight')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight distribution histogram
    ax2.hist(proximity_labels.cpu().numpy(), bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Continuous Weight')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Continuous Weights')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_retrocue_allocation_gif(
    history_surfaces: List[np.ndarray],
    history_ind_pts: List[np.ndarray],
    items: List[Tuple],
    cued_item_idx: int,
    cue_start_epoch: int,
    save_dir: str = "visualizations",
    filename: str = "retrocue_allocation.gif",
) -> None:
    """
    Animated GIF showing how computational resources (inducing points + predictive
    mean) dynamically shift toward the cued item during the maintenance phase.

    Layout — two side-by-side panels per frame:
      Left:  2D GP surface heatmap with inducing points overlaid.
             Cued item is highlighted with a coloured star + vertical/horizontal
             dashed lines so you can track whether the mean peak rises there.
      Right: 1D marginal — mean activation averaged over all colors at each
             *location* value. A vertical line marks the cued location so you
             can see the peak drifting toward it as the cue takes effect.

    Parameters
    ----------
    history_surfaces : list of (res, res) np.ndarray
        Predictive mean surfaces recorded each epoch.
    history_ind_pts  : list of (M, 2) np.ndarray
        Inducing point locations recorded each epoch.
    items            : list of (loc, color) tuples
        Ground-truth item positions.
    cued_item_idx    : int
        Which item is being cued (highlighted in red).
    cue_start_epoch  : int
        Epoch at which the retrocue begins — annotated with a marker in the title.
    save_dir / filename : str
        Output path.
    """
    print("Creating retrocue allocation GIF...")
    if not history_surfaces:
        return

    res = history_surfaces[0].shape[0]
    locs_np = np.linspace(-180.0, 180.0, res)

    cued_loc   = items[cued_item_idx][0]
    cued_color = items[cued_item_idx][1]
    item_rgb   = _item_colors_from_wheel([it[1] for it in items])

    global_vmax = np.max([np.max(s) for s in history_surfaces])
    global_vmin = np.min([np.min(s) for s in history_surfaces])

    # --- build 1-D color slice at the cued location row ---
    # surf shape: (res_loc, res_color); axis 0 = location, axis 1 = color
    # find the row index closest to cued_loc
    locs_np   = np.linspace(-180.0, 180.0, res)
    colors_np = np.linspace(-180.0, 180.0, res)
    cued_loc_idx = int(np.argmin(np.abs(locs_np - cued_loc)))
    marginals = [surf[cued_loc_idx, :]  for surf in history_surfaces]  # (res,) each

    fig, (ax_map, ax_marg) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: 2-D surface ──────────────────────────────────────────────
    im = ax_map.imshow(
        history_surfaces[0].T,
        extent=[-180.0, 180.0, -180.0, 180.0],
        origin='lower', cmap='viridis', aspect='auto',
        vmin=global_vmin, vmax=global_vmax,
    )
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label('Predictive Mean', fontsize=10)

    # static items
    for i, (il, ic) in enumerate(items):
        if i == cued_item_idx:
            ax_map.scatter(il, ic, c=[item_rgb[i]], s=350, marker='*',
                           edgecolors='white', linewidths=2.0, zorder=6, label='Cued item')
            ax_map.axvline(il, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
            ax_map.axhline(ic, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
        else:
            ax_map.scatter(il, ic, c=[item_rgb[i]], s=200, marker='*',
                           edgecolors='white', linewidths=0.8, zorder=6)

    # dynamic inducing points
    sc_ind = ax_map.scatter(
        history_ind_pts[0][:, 0], history_ind_pts[0][:, 1],
        c='white', marker='.', s=40, alpha=0.6, zorder=5, label='Inducing pts',
    )
    ax_map.set_xlim(-180, 180); ax_map.set_ylim(-180, 180)
    ax_map.set_xlabel('Location (°)', fontsize=11)
    ax_map.set_ylabel('Color (°)', fontsize=11)
    ax_map.legend(loc='upper right', fontsize=8)
    title_map = ax_map.set_title('', fontsize=12)

    # ── Right panel: color slice at cued location ───────────────────────────
    line_marg, = ax_marg.plot(colors_np, marginals[0], color='#2196F3', linewidth=2.5)
    ax_marg.set_xlim(-180, 180)
    marg_vmin = min(m.min() for m in marginals)
    marg_vmax = max(m.max() for m in marginals)
    ax_marg.set_ylim(marg_vmin - 0.05*(marg_vmax-marg_vmin),
                     marg_vmax + 0.10*(marg_vmax-marg_vmin))
    ax_marg.axvline(cued_color, color=_item_colors_from_wheel([cued_color])[0],
                    linestyle='--', linewidth=1.8, alpha=0.9,
                    label=f'True color ({cued_color:.0f}°)')
    ax_marg.set_xlabel('')
    ax_marg.set_xticks([])
    ax_marg.set_ylabel(f'Mean activation\n@ loc={cued_loc:.0f}°', fontsize=11)
    ax_marg.legend(loc='upper right', fontsize=8)
    title_marg = ax_marg.set_title('', fontsize=12)
    ax_marg.grid(True, linestyle='--', alpha=0.4)

    # ── colorwheel strip below ax_marg ───────────────────────────────────────
    cw = _load_colorwheel()                        # (360, 3)
    cw_img = cw[np.newaxis, :, :]                  # (1, 360, 3)
    divider = make_axes_locatable(ax_marg)
    ax_cw = divider.append_axes('bottom', size='8%', pad=0.0)
    ax_cw.imshow(cw_img, aspect='auto', extent=[-180, 180, 0, 1],
                 origin='lower', interpolation='bilinear')
    ax_cw.set_xlim(-180, 180)
    ax_cw.set_xticks([-180, -90, 0, 90, 180])
    ax_cw.set_yticks([])
    ax_cw.set_xlabel('Color (°)', fontsize=11)
    ax_cw.axvline(x=cued_color, color=_item_colors_from_wheel([cued_color])[0],
                  linestyle='--', linewidth=2.0)

    fig.tight_layout(pad=2.0)

    def update(frame):
        phase = 'Pre-cue' if frame < cue_start_epoch else '▶ CUE ACTIVE'
        epoch_label = f'Maint epoch {frame}  |  {phase}'

        # Update surface
        im.set_array(history_surfaces[frame].T)
        sc_ind.set_offsets(history_ind_pts[frame])
        title_map.set_text(f'GP Surface  —  {epoch_label}')

        # Update marginal
        line_marg.set_ydata(marginals[frame])
        title_marg.set_text(f'Color slice @ loc={cued_loc:.0f}°  —  {epoch_label}')
        return im, sc_ind, line_marg

    anim = FuncAnimation(fig, update, frames=len(history_surfaces),
                         interval=120, blit=False)
    os.makedirs(save_dir, exist_ok=True)
    anim.save(os.path.join(save_dir, filename), dpi=100, writer='pillow')
    plt.close()
    print(f"  Saved → {os.path.join(save_dir, filename)}")


def plot_retrocue_allocation_comparison(
    surface_pre: np.ndarray,
    ind_pts_pre: np.ndarray,
    surface_post: np.ndarray,
    ind_pts_post: np.ndarray,
    items: List[Tuple],
    cued_item_idx: int,
    save_dir: str = "visualizations",
    filename: str = "retrocue_allocation_comparison.png",
) -> None:
    """
    Static 2×2 before/after comparison showing retrocue-driven resource reallocation.

    Rows: Before cue onset  |  After cue onset
    Cols: 2-D GP surface heatmap  |  Location marginal (1-D)

    The cued item is highlighted in red throughout. Inducing points are shown
    in each heatmap panel so you can directly see whether they have drifted
    toward the cued location after the cue takes effect.

    Parameters
    ----------
    surface_pre / surface_post  : (res, res) np.ndarray
        Predictive mean surface snapshots *before* and *after* cue onset.
    ind_pts_pre / ind_pts_post  : (M, 2) np.ndarray
        Inducing point locations at the two time-points.
    items              : list of (loc, color) tuples
    cued_item_idx      : int
    save_dir / filename : str
    """
    print("Plotting retrocue allocation comparison...")
    res = surface_pre.shape[0]
    locs_np   = np.linspace(-180.0, 180.0, res)
    colors_np = np.linspace(-180.0, 180.0, res)

    cued_loc   = items[cued_item_idx][0]
    cued_color = items[cued_item_idx][1]
    item_rgb   = _item_colors_from_wheel([it[1] for it in items])

    global_vmax = float(np.max([surface_pre.max(), surface_post.max()]))
    global_vmin = float(np.min([surface_pre.min(), surface_post.min()]))

    # Color slice at the nearest row to cued_loc
    cued_loc_idx = int(np.argmin(np.abs(locs_np - cued_loc)))
    marg_pre  = surface_pre [cued_loc_idx, :]
    marg_post = surface_post[cued_loc_idx, :]
    marg_vmax = max(marg_pre.max(), marg_post.max())
    marg_vmin = min(marg_pre.min(), marg_post.min())
    marg_pad  = 0.08 * (marg_vmax - marg_vmin)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    row_labels = ['Before cue onset', 'After cue onset']
    surfaces   = [surface_pre,  surface_post]
    ind_pts_list = [ind_pts_pre, ind_pts_post]
    marginals  = [marg_pre,   marg_post]
    line_colors = ['steelblue', 'darkorange']

    for row_idx, (label, surf, ind_pts, marg) in enumerate(
        zip(row_labels, surfaces, ind_pts_list, marginals)
    ):
        ax_heat = axes[row_idx][0]
        ax_marg = axes[row_idx][1]

        # ── Heatmap ──────────────────────────────────────────────────────────
        im = ax_heat.imshow(
            surf.T,
            extent=[-180.0, 180.0, -180.0, 180.0],
            origin='lower', cmap='viridis', aspect='auto',
            vmin=global_vmin, vmax=global_vmax,
        )
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label('Predictive Mean', fontsize=9)

        # Inducing points
        ax_heat.scatter(ind_pts[:, 0], ind_pts[:, 1],
                        c='white', marker='.', s=35, alpha=0.55,
                        zorder=5, label='Inducing pts')

        # Items
        for i, (il, ic) in enumerate(items):
            if i == cued_item_idx:
                ax_heat.scatter(il, ic, c=[item_rgb[i]], s=400, marker='*',
                                edgecolors='white', linewidths=2.2, zorder=7,
                                label='Cued item')
                ax_heat.axvline(il, color='white', linestyle='--',
                                linewidth=1.3, alpha=0.7)
                ax_heat.axhline(ic, color='white', linestyle='--',
                                linewidth=1.3, alpha=0.7)
            else:
                ax_heat.scatter(il, ic, c=[item_rgb[i]], s=220, marker='*',
                                edgecolors='white', linewidths=0.8, zorder=7)

        ax_heat.set_xlim(-180, 180); ax_heat.set_ylim(-180, 180)
        ax_heat.set_title(f'{label}  —  GP Surface', fontsize=12, pad=8)
        ax_heat.set_xlabel('Location (°)', fontsize=10)
        ax_heat.set_ylabel('Color (°)', fontsize=10)
        ax_heat.legend(loc='upper right', fontsize=8)

        # ── Color slice at cued location ──────────────────────────────────────
        ax_marg.plot(colors_np, marg, color='#2196F3', linewidth=2.8)

        ax_marg.axvline(cued_color, color=_item_colors_from_wheel([cued_color])[0],
                        linestyle='--', linewidth=1.8, alpha=0.9,
                        label=f'True color ({cued_color:.0f}°)')

        peak_col = colors_np[np.argmax(marg)]
        ax_marg.axvline(peak_col, color=_item_colors_from_wheel([peak_col])[0],
                        linestyle=':', linewidth=1.8, alpha=0.9,
                        label=f'Peak ({peak_col:.0f}°)')

        ax_marg.set_xlim(-180, 180)
        ax_marg.set_ylim(marg_vmin - marg_pad, marg_vmax + 2*marg_pad)
        ax_marg.set_title(f'{label}  —  Color slice @ loc={cued_loc:.0f}°', fontsize=12, pad=8)
        ax_marg.set_xlabel('')
        ax_marg.set_xticks([])
        ax_marg.set_ylabel(f'Mean activation\n@ loc={cued_loc:.0f}°', fontsize=10)
        ax_marg.legend(loc='upper right', fontsize=8)
        ax_marg.grid(True, linestyle='--', alpha=0.4)

        # ── colorwheel strip below ax_marg ────────────────────────────────────
        cw = _load_colorwheel()                        # (360, 3)
        cw_img = cw[np.newaxis, :, :]                  # (1, 360, 3)
        divider = make_axes_locatable(ax_marg)
        ax_cw = divider.append_axes('bottom', size='8%', pad=0.0)
        ax_cw.imshow(cw_img, aspect='auto', extent=[-180, 180, 0, 1],
                     origin='lower', interpolation='bilinear')
        ax_cw.set_xlim(-180, 180)
        ax_cw.set_xticks([-180, -90, 0, 90, 180])
        ax_cw.set_yticks([])
        ax_cw.set_xlabel('Color (°)', fontsize=10)
        ax_cw.axvline(x=cued_color, color=_item_colors_from_wheel([cued_color])[0],
                      linestyle='--', linewidth=2.0)
        ax_cw.axvline(x=peak_col,   color=_item_colors_from_wheel([peak_col])[0],
                      linestyle=':',  linewidth=2.0)

    fig.suptitle('Retrocue-Driven Resource Reallocation',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2.5)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {os.path.join(save_dir, filename)}")


if __name__ == "__main__":
    fig = create_3d_adaptive_computation()
    plt.show()
