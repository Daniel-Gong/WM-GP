import os
import sys

# This file lives under src/viz/; add src/ so generator and attention_mechanisms resolve.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from generator import generate_items
from attention_mechanisms import SpatialProximityAttention

def plot_attention_mechanism():
    """
    Illustrates how spatial proximity attention dynamically allocates 
    computational resources (gradient weights on the maintenance grid) 
    in the GP model.
    """
    plt.style.use('default')
    # Use a wider layout to accommodate the colorbar and keep the main plot square
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # 1. Define memory items
    # (Location, Color)
    items = generate_items(4, seed=42)
    cued_idx = 0
    cued_location = items[cued_idx][0]
    
    # 2. Recreate the maintenance grid from simulation.py
    maint_grid_size = 30  # Actual size used in simulation.py
    grid_1d = np.linspace(-180.0, 180.0, maint_grid_size + 1)[:-1]
    # Center points in bins
    bin_width = 360.0 / maint_grid_size
    grid_1d = grid_1d + bin_width / 2.0
    
    grid_loc, grid_color = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    
    maint_loc_flat = grid_loc.flatten()
    maint_color_flat = grid_color.flatten()
    
    # 3. Calculate attention weights using exactly our model's mechanism
    spatial_std = 30.0
    attended_gain = 3.0
    
    attn = SpatialProximityAttention(spatial_std=spatial_std, attended_gain=attended_gain)
    
    loc_tensor = torch.tensor(maint_loc_flat, dtype=torch.float32)
    weights_tensor = attn(loc_tensor, float(cued_location))
    resource_weights = weights_tensor.numpy()
    
    # 4. Create color map for resource allocation
    min_weight = 0.5  # Neutral baseline from code
    max_weight = attended_gain
    
    # Normalize for colormap
    norm_weights = (resource_weights - min_weight) / (max_weight - min_weight)
    colors = plt.cm.viridis(norm_weights)
    
    # Setup axes
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.set_aspect('equal')
    
    # 5. Draw maintenance grid points with size proportional to resource allocation
    # Base size of the circle patch
    max_radius = bin_width * 0.45
    min_radius = bin_width * 0.15
    
    for i in range(len(maint_loc_flat)):
        # Size proportional to weight
        size = min_radius + (max_radius - min_radius) * norm_weights[i]
        circle = Circle((maint_loc_flat[i], maint_color_flat[i]), size, 
                        color=colors[i], alpha=0.7, zorder=2)
        ax1.add_patch(circle)
        
    # 6. Mark spatial distances with vertical bands
    # Because attention is 1D on Location, distance is along X axis
    for num_std in [1, 2, 3]:
        dist = num_std * spatial_std
        alpha_val = 0.5 - 0.1 * num_std
        
        for sign in [-1, 1]:
            x_pos = cued_location + sign * dist
            
            # Wrap around manually
            while x_pos > 180: x_pos -= 360
            while x_pos < -180: x_pos += 360
                
            ax1.axvline(x=x_pos, color='gray', linestyle='--', alpha=alpha_val, zorder=1)
            
            if sign == 1 and num_std <= 2:
                y_text = 170
                ax1.text(x_pos + 3, y_text, f'{num_std}σ\n(d={dist:.0f}°)', 
                         fontsize=10, va='top', ha='left', color='gray', 
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0), zorder=3)
                 
    # 7. Draw the memorized items
    for i, (loc, col) in enumerate(items):
        is_cued = (i == cued_idx)
        marker_color = 'red' if is_cued else 'white'
        edge_color = 'white' if is_cued else 'black'
        linewidth = 2 if is_cued else 1
        
        # Plot item
        ax1.plot(loc, col, marker='*', markersize=20 if is_cued else 15, 
                markerfacecolor=marker_color, markeredgecolor=edge_color, 
                markeredgewidth=linewidth, zorder=5)
        
        # Label
        label = 'Cued Item' if is_cued else f'Item {i+1}'
        text_color = 'red' if is_cued else 'black'
        ax1.text(loc, col - 15, label, 
                ha='center', va='top', fontsize=12, fontweight='bold', color=text_color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=6)
                
    # Emphasize the cued location's spatial coordinate (1D attention)
    ax1.axvline(x=cued_location, color='red', linestyle='-', alpha=0.4, linewidth=3, zorder=1)
    
    # Formatting
    ax1.set_title('Adaptive Computation on Maintenance Grid\n(Spatial Proximity Attention)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Location (°)', fontsize=16)
    ax1.set_ylabel('Color (°)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_yticks([-180, -90, 0, 90, 180])
    
    # Add a colorbar for the weights
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    # Pad allows making room for the colorbar without squishing the aspect ratio
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (Resource Allocation)', fontsize=14)
    
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations', 'attention_demo')
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'adaptive_computation.png')
    pdf_path = os.path.join(out_dir, 'adaptive_computation.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    
if __name__ == "__main__":
    plot_attention_mechanism()
