import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import gpytorch
from typing import Dict, List, Tuple
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_gp_surface_2d(model, likelihood, items, epoch, prefix="", save_dir="visualizations", filename="gp_surface_final.png"):
    """
    Plots a 2D heatmap of the GP predictive Mean with Inducing Points overlaid.
    X-axis: Location (-pi, pi)
    Y-axis: Color (-pi, pi)
    """
    model.eval()
    likelihood.eval()
    
    res = 50
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
    
    # Plot true items
    true_locs = [i[0] for i in items]
    true_cols = [i[1] for i in items]
    ax.scatter(true_locs, true_cols, c='r', marker='*', s=200, label='Encoded Items', edgecolors='white')
    
    # Plot inducing points
    if hasattr(model, 'variational_strategy'):
        ind_pts = model.variational_strategy.inducing_points.detach().cpu().numpy()
        ax.scatter(ind_pts[:, 0], ind_pts[:, 1], c='white', marker='.', s=50, alpha=0.5, label='Inducing Pts')
        
    ax.set_title(f"{prefix} GP Surface - Epoch {epoch}")
    ax.set_xlabel("Location (deg)")
    ax.set_ylabel("Color (deg)")
    ax.set_xlim([-180.0, 180.0])
    ax.set_ylim([-180.0, 180.0])
    ax.legend(loc='upper right')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def plot_training_trajectories(history: Dict, save_dir="visualizations",filename="training_trajectories.png"):
    """
    Plots the trajectories of GP Hyperparameters (Lengthscales, Noise) 
    and ELBO Losses over the training epochs.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Encoding Loss
    axs[0, 0].plot(history.get('encoding_loss', []), color='blue', label='Encoding (-ELBO)')
    axs[0, 0].set_title("Variational ELBO (Encoding)")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    
    # Plot 2: Maintenance Loss (if any)
    if 'maintenance_loss' in history and len(history['maintenance_loss']) > 0:
        axs[0, 1].plot(history['maintenance_loss'], color='orange', label='Maint (KL + Attn)')
        axs[0, 1].set_title("Maintenance Loss")
    else:
        axs[0, 1].text(0.5, 0.5, 'No Maintenance Run', ha='center')
    axs[0, 1].set_xlabel("Epoch")
    
    # Plot 3: Specific Model Parameter Trajectories
    epochs = len(history.get('loc_lengthscale', []))
    if epochs > 0:
        axs[1, 0].plot(history['loc_lengthscale'], label='Location LS', color='red')
        axs[1, 0].plot(history['color_lengthscale'], label='Color LS', color='green')
        axs[1, 0].set_title("Periodic Lengthscales")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].legend()

    plt.tight_layout()
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
    ax.scatter(true_locs, true_cols, c='r', marker='*', s=200, label='Items', edgecolors='white', zorder=5)
    
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

def create_gp_surface_3d_gif(history_surfaces: List[np.ndarray], history_ind_pts: List[np.ndarray], history_ind_vals: List[np.ndarray], items: List[Tuple], save_dir="visualizations", filename="gp_optimization_3d.gif"):
    """
    Animates a 3D surface plot of the GP along with floating inducing points over epochs.
    Requires history_surfaces to contain eval-grid (50x50) mean outputs.
    """
    if not history_surfaces:
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 50x50 grid that matches the eval grid in simulation.py
    res = 50
    locs = np.linspace(-180.0, 180.0, res)
    colors = np.linspace(-180.0, 180.0, res)
    L, C = np.meshgrid(locs, colors, indexing='ij')
    
    # Determine global Z limits
    z_max = max(1.0, np.max([np.max(s) for s in history_surfaces]))
    z_min = np.min([np.min(s) for s in history_surfaces])
    
    def update(frame):
        ax.clear()
        ax.set_title(f"3D GP Optimization - Epoch {frame}")
        ax.set_xlabel("Location (deg)")
        ax.set_ylabel("Color (deg)")
        ax.set_zlabel("Predictive Mean")
        ax.set_xlim([-180.0, 180.0])
        ax.set_ylim([-180.0, 180.0])
        ax.set_zlim([z_min, z_max])
        
        # Plot surface
        surf = ax.plot_surface(L, C, history_surfaces[frame], cmap='viridis', edgecolor='none', alpha=0.7)
        
        # Plot inducing points in 3D
        ind_pts = history_ind_pts[frame]
        ind_vals = history_ind_vals[frame]
        ax.scatter(ind_pts[:, 0], ind_pts[:, 1], ind_vals, color='red', s=50, label='Ind. Pts', depthshade=False)
        
        # Plot items at Z=0 for reference
        true_locs = [i[0] for i in items]
        true_cols = [i[1] for i in items]
        ax.scatter(true_locs, true_cols, [z_min]*len(items), c='cyan', marker='*', s=200, label='Items (Ground)', edgecolors='black')
        
        ax.legend(loc='upper right')
        return fig,

    anim = FuncAnimation(fig, update, frames=len(history_surfaces), interval=100, blit=False)
    
    os.makedirs(save_dir, exist_ok=True)
    anim.save(os.path.join(save_dir, filename), dpi=100, writer='pillow')
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

    # ── 1. Individual histograms per set size ────────────────────────────────
    for ss in set_sizes:
        errs = np.array(errors_per_set_size[ss])
        sd   = np.std(errs)

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(-180.0, 180.0, 37)          # 10-deg bins
        ax.hist(errs, bins=bins, color='steelblue', edgecolor='black', alpha=0.75, density=True)

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
        _vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(_vis_dir, exist_ok=True)
        plt.savefig(os.path.join(_vis_dir, f"error_dist_N{ss}.png"), dpi=150)
        plt.close()
        print(f"  Saved error distribution for N={ss}  (SD={sd:.2f}°)")

    # ── 2. Combined panel (all set sizes side by side) ───────────────────────
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    bins = np.linspace(-180.0, 180.0, 37)
    for idx, ss in enumerate(set_sizes):
        row, col = divmod(idx, ncols)
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

def plot_retrocue_benefit(neutral_mae: float, cued_mae: float, set_size: int, save_dir="visualizations"):
    """
    Plots a bar chart comparing Neutral vs Cued Mean Absolute Errors.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    conditions = ['Neutral', 'Cued']
    maes = [neutral_mae, cued_mae]
    colors = ['lightcoral', 'lightgreen']
    
    ax.bar(conditions, maes, color=colors, edgecolor='black', width=0.5)
    ax.set_title(f"Retrocue Benefit (N={set_size})", fontsize=14)
    ax.set_ylabel("Mean Absolute Error (deg)", fontsize=12)
    
    # Annotate bars
    for i, v in enumerate(maes):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"retrocue_benefit_N{set_size}.png"), dpi=150)
    plt.close()

def plot_bias_effect(df: pd.DataFrame, save_dir="visualizations"):
    """
    Plots the Bias curve across color feature distances.
    Assumes Bias > 0 is Attraction (pulled towards distractor offset), Bias < 0 is Repulsion.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    dist_deg = df['Distance_deg']
    bias_deg = df['Bias_deg']
    
    ax.plot(dist_deg, bias_deg, marker='o', linestyle='-', color='purple', markersize=8)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    ax.set_title("Attraction/Repulsion Bias vs Featural Distance", fontsize=14)
    ax.set_xlabel("Target-Distractor Distance (degrees)", fontsize=12)
    ax.set_ylabel("Bias Magnitude (degrees)\n(+) Attraction, (-) Repulsion", fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "bias_effect.png"), dpi=150)
    plt.close()

def visualize_training_results(samples, weights, encoding_losses, maintenance_losses, inducing_points_history, inducing_values_history, retrieval_errors):
    """
    Visualize the training results, including:
    1. Training loss curve
    2. Animation of inducing points and their values during training, with trajectory traces
    3. Sample data distribution in the background
    4. Item representation metrics over time
    """
    # Create figure with subplots using gridspec for custom widths
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 5)
    
    # Plot encoding loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(encoding_losses)
    ax1.set_title('Encoding Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_box_aspect(1)
    
    # Plot maintenance loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(maintenance_losses)
    ax2.set_title('Maintenance Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_box_aspect(1)
    
    # Plot decision loss
    ax3 = fig.add_subplot(gs[0:2, 1:3])
    for item_idx in range(len(retrieval_errors)):
        ax3.plot(retrieval_errors[item_idx], 
                label=f'Item {item_idx}', alpha=0.7)
    ax3.set_title('Retrieval Error Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Retrieval Error')
    ax3.legend()
    ax3.set_box_aspect(1)
    
    # Create 3D subplot for inducing points animation
    ax4 = fig.add_subplot(gs[0:2, 3:5], projection='3d')
    
    # Plot sample data in the background using tensor data directly
    ax4.scatter(samples[:, 0].detach().cpu().numpy(), 
                samples[:, 1].detach().cpu().numpy(), 
                samples[:, 2].detach().cpu().numpy(),
                c=weights.detach().cpu().numpy(), cmap='viridis', alpha=0.1, s=10, label='Samples')
    
    # Initialize scatter plot for current points
    scatter = ax4.scatter([], [], [], c=[], cmap='viridis', alpha=0.8, s=10)
    
    # Initialize line collections for trajectories
    num_points = inducing_points_history[0].shape[0]
    lines = [ax4.plot([], [], [], alpha=0.3, linewidth=1)[0] for _ in range(num_points)]
    
    plt.colorbar(scatter, ax=ax4, label='Function Value')
    
    # Set labels and limits
    ax4.set_xlabel('Location X')
    ax4.set_ylabel('Location Y')
    ax4.set_zlabel('Color')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.set_title('Inducing Points Evolution')
    ax4.legend()
    
    def update(frame):
        # Get inducing points and values for current frame
        points = inducing_points_history[frame]
        values = inducing_values_history[frame]
        
        # Update scatter plot
        scatter._offsets3d = (points[:, 0].detach().cpu().numpy(), 
                            points[:, 1].detach().cpu().numpy(), 
                            points[:, 2].detach().cpu().numpy())
        scatter.set_array(values.detach().cpu().numpy())
        
        # Update trajectory lines
        for i in range(num_points):
            # Get trajectory up to current frame
            trajectory = torch.stack([p[i].detach() for p in inducing_points_history[:frame+1]])
            trajectory_values = torch.stack([v[i].detach() for v in inducing_values_history[:frame+1]])
            
            # Update line data
            lines[i].set_data(trajectory[:, 0].cpu().numpy(), trajectory[:, 1].cpu().numpy())
            lines[i].set_3d_properties(trajectory[:, 2].cpu().numpy())
            
            # Set line color based on function values
            lines[i].set_color(plt.cm.viridis(trajectory_values[-1].cpu().numpy()))

        # Update title with epoch number
        ax4.set_title(f'Inducing Points Evolution (Epoch {frame})')
        
        return [scatter] + lines
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(inducing_points_history),
        interval=100, blit=False
    )

    plt.tight_layout()
    plt.show()

    return anim

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

def create_3d_adaptive_computation():
    """Create 3D illustration of adaptive computation with location cueing and color retrieval"""
    fig = plt.figure(figsize=(20, 16))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the 3D space
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Generate sample items in 3D space (x, y, color)
    np.random.seed(42)
    n_items = 8
    items = np.random.uniform(0.1, 0.9, (n_items, 3))
    
    # Cued location (we'll cue the location of item 0)
    cued_item_idx = 0
    cued_location = items[cued_item_idx, :2]  # x, y coordinates
    cued_color = items[cued_item_idx, 2]      # color value
    
    # Calculate distances from cued location (only x,y coordinates)
    distances_2d = np.linalg.norm(items[:, :2] - cued_location, axis=1)
    
    # Resource allocation based on 2D distance
    resource_weights = 0.95 * np.exp(-distances_2d / 0.2) + 0.05
    resource_weights = resource_weights / np.sum(resource_weights)
    
    # Create color map for resource allocation
    colors_3d = plt.cm.viridis(resource_weights / max(resource_weights))
    
    # Plot items in 3D space
    for i, (item, weight) in enumerate(zip(items, resource_weights)):
        x, y, color = item
        
        # Size based on resource allocation
        size = 0.02 + 0.08 * weight / max(resource_weights)
        
        # Plot item as a sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = x + size * np.outer(np.cos(u), np.sin(v))
        y_sphere = y + size * np.outer(np.sin(u), np.sin(v))
        z_sphere = color + size * np.outer(np.ones_like(u), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                       color=colors_3d[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add item labels
        ax.text(x, y, color + size + 0.05, f'Item {i}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add resource allocation values
        ax.text(x, y, color - size - 0.05, f'w={weight:.2f}', 
                ha='center', va='top', fontsize=8)
    
    # Highlight cued item
    x_cued, y_cued, color_cued = items[cued_item_idx]
    size_cued = 0.02 + 0.08 * resource_weights[cued_item_idx] / max(resource_weights)
    
    # Create larger, highlighted sphere for cued item
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = x_cued + size_cued * np.outer(np.cos(u), np.sin(v))
    y_sphere = y_cued + size_cued * np.outer(np.sin(u), np.sin(v))
    z_sphere = color_cued + size_cued * np.outer(np.ones_like(u), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                   color='red', alpha=0.9, edgecolor='white', linewidth=2)
    
    # Add attention spotlight (cylinder around cued location)
    spotlight_radius = 0.3
    spotlight_height = 1.0
    z_spotlight = np.linspace(0, 1, 20)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_spotlight)
    
    x_spotlight = x_cued + spotlight_radius * np.cos(theta_grid)
    y_spotlight = y_cued + spotlight_radius * np.sin(theta_grid)
    
    ax.plot_surface(x_spotlight, y_spotlight, z_grid, 
                   color='red', alpha=0.2, edgecolor='red', linewidth=1)
    
    # Add cue arrow pointing to cued location
    cue_start = np.array([0.1, 0.1, 0.9])
    cue_end = np.array([x_cued, y_cued, color_cued])
    cue_direction = cue_end - cue_start
    cue_length = np.linalg.norm(cue_direction)
    cue_direction = cue_direction / cue_length * 0.8
    
    ax.quiver(cue_start[0], cue_start[1], cue_start[2],
              cue_direction[0], cue_direction[1], cue_direction[2],
              color='red', arrow_length_ratio=0.2, linewidth=3, alpha=0.8)
    
    ax.text(cue_start[0], cue_start[1], cue_start[2] + 0.1, 'Cue\nLocation', 
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
    
    # Add retrieval arrow from cued location to color axis
    retrieval_start = np.array([x_cued, y_cued, color_cued])
    retrieval_end = np.array([0.1, 0.1, color_cued])
    retrieval_direction = retrieval_end - retrieval_start
    retrieval_length = np.linalg.norm(retrieval_direction)
    retrieval_direction = retrieval_direction / retrieval_length * 0.6
    
    ax.quiver(retrieval_start[0], retrieval_start[1], retrieval_start[2],
              retrieval_direction[0], retrieval_direction[1], retrieval_direction[2],
              color='blue', arrow_length_ratio=0.2, linewidth=3, alpha=0.8)
    
    ax.text(0.05, 0.05, color_cued, f'Retrieved\nColor: {color_cued:.2f}', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='blue',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Add coordinate axes with labels
    ax.set_xlabel('Location X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Location Y', fontsize=14, fontweight='bold')
    ax.set_zlabel('Color', fontsize=14, fontweight='bold')
    
    # Add title
    ax.set_title('3D Adaptive Computation: Location Cueing → Color Retrieval', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add mathematical formulation
    formula_text = 'L_adapt = 0.95 * L_cued + 0.05 * L_non-cued'
    ax.text2D(0.5, 0.02, formula_text, ha='center', va='center', fontsize=14, fontweight='bold',
              transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.8, label='Cued Item'),
        mpatches.Patch(color='blue', alpha=0.8, label='Other Items'),
        mpatches.Patch(color='red', alpha=0.2, label='Attention Spotlight')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    return fig

def create_3d_gp_visualization():
    """Create 3D visualization of GP function over the input space"""
    fig = plt.figure(figsize=(20, 16))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the 3D space
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Create grid for GP surface
    x_grid = np.linspace(0, 1, 20)
    y_grid = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create a sample GP function (color as function of location)
    # This represents the learned mapping from location to color
    Z = 0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.2 * np.random.normal(0, 0.1, X.shape)
    Z = np.clip(Z, 0, 1)  # Clip to valid color range
    
    # Plot GP surface
    surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis', linewidth=0)
    
    # Add inducing points
    inducing_points = np.array([
        [0.2, 0.3, 0.6],
        [0.5, 0.5, 0.5],
        [0.8, 0.2, 0.4],
        [0.3, 0.8, 0.7],
        [0.7, 0.7, 0.3]
    ])
    
    for i, point in enumerate(inducing_points):
        x, y, z = point
        ax.scatter(x, y, z, c='red', s=100, alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(x, y, z + 0.05, f'IP{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add sample items
    items = np.array([
        [0.3, 0.4, 0.6],
        [0.6, 0.3, 0.4],
        [0.4, 0.7, 0.7],
        [0.8, 0.6, 0.3]
    ])
    
    for i, item in enumerate(items):
        x, y, z = item
        ax.scatter(x, y, z, c='blue', s=80, alpha=0.8, edgecolor='white', linewidth=2)
        ax.text(x, y, z + 0.05, f'Item{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add cued location and retrieval
    cued_location = np.array([0.5, 0.5])
    cued_x, cued_y = cued_location
    
    # Find color at cued location using GP
    cued_color = 0.5 + 0.3 * np.sin(2 * np.pi * cued_x) * np.cos(2 * np.pi * cued_y)
    cued_color = np.clip(cued_color, 0, 1)
    
    # Plot cued location
    ax.scatter(cued_x, cued_y, cued_color, c='red', s=200, alpha=0.9, edgecolor='white', linewidth=3)
    
    # Add retrieval arrow to color axis
    retrieval_start = np.array([cued_x, cued_y, cued_color])
    retrieval_end = np.array([0.1, 0.1, cued_color])
    retrieval_direction = retrieval_end - retrieval_start
    retrieval_length = np.linalg.norm(retrieval_direction)
    retrieval_direction = retrieval_direction / retrieval_length * 0.6
    
    # Add coordinate axes with labels
    ax.set_xlabel('Location X', fontsize=20, fontweight='bold')
    ax.set_ylabel('Location Y', fontsize=20, fontweight='bold')
    ax.set_zlabel('Color', fontsize=20, fontweight='bold')
    
    # Add title
    ax.set_title('3D Gaussian Process: Location → Color Mapping', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.8)
    cbar.set_label('Function Value', fontsize=20,rotation=270)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    return fig

