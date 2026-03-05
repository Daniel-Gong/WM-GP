import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from typing import List, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_items(
    n_items: int,
    seed: Optional[int] = None,
    n_slots: int = 8,
) -> List[Tuple[float, float]]:
    """
    Generate items with (location, color) coordinates bounded [-180, 180).

    Locations are placed on ``n_slots`` equally-spaced positions arranged on
    an invisible circle (spacing = 360 / n_slots degrees).  A single random
    rotation is applied so the slot grid is different on every trial.
    ``n_items`` slots are drawn without replacement from the ``n_slots``
    available positions.

    Parameters
    ----------
    n_items : int
        Number of items to generate (must be <= n_slots).
    seed : int, optional
        Random seed for reproducibility.
    n_slots : int
        Total number of equally-spaced slots on the circle (default 8).

    Returns
    -------
    items : list of (location, color) tuples in degrees [-180, 180)
    """
    if n_items > n_slots:
        raise ValueError(f"n_items ({n_items}) must be <= n_slots ({n_slots})")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Build the n_slots equally-spaced positions then apply a random rotation
    spacing = 360.0 / n_slots
    rotation = np.random.uniform(0.0, spacing)          # random offset within one slot width
    slot_positions = [(i * spacing + rotation) % 360.0 - 180.0 for i in range(n_slots)]

    # Choose n_items slots without replacement and assign random colors
    chosen_indices = np.random.choice(n_slots, size=n_items, replace=False)
    locs   = [slot_positions[i] for i in chosen_indices]
    colors = [np.random.uniform(-180.0, 180.0) for _ in range(n_items)]

    return [(locs[i], colors[i]) for i in range(n_items)]

def sample_training_data(
    items: List[Tuple[float, float]],
    n_samples_per_item: int,
    loc_std: float,
    color_std: float,
    loc_encoding_noise_std: float = 0.0,
    color_encoding_noise_std: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training samples around each 1D item.
    """
    all_samples = []
    all_weights = []
    all_ids = []
    
    for item_idx, (l, c) in enumerate(items):
        # Apply encoding noise (trial-to-trial drift)
        l_encoded = l + np.random.normal(0, loc_encoding_noise_std) if loc_encoding_noise_std > 0 else l
        c_encoded = c + np.random.normal(0, color_encoding_noise_std) if color_encoding_noise_std > 0 else c
        
        # Wrap encoded to [-180, 180)
        l_encoded = (l_encoded + 180.0) % 360.0 - 180.0
        c_encoded = (c_encoded + 180.0) % 360.0 - 180.0
        
        # Sample noisy observations
        locs = np.random.normal(l_encoded, loc_std, n_samples_per_item)
        colors = np.random.normal(c_encoded, color_std, n_samples_per_item)
        
        # Wrap observations
        locs = (locs + 180.0) % 360.0 - 180.0
        colors = (colors + 180.0) % 360.0 - 180.0
        
        samples = np.column_stack([locs, colors])
        
        # Compute weights based on CIRCULAR distance
        loc_dist = np.minimum(np.abs(locs - l_encoded), 360.0 - np.abs(locs - l_encoded))
        color_dist = np.minimum(np.abs(colors - c_encoded), 360.0 - np.abs(colors - c_encoded))
        
        loc_weight = np.exp(-0.5 * (loc_dist / loc_std)**2)
        color_weight = np.exp(-0.5 * (color_dist / color_std)**2)
        weights = loc_weight * color_weight
        weights = weights / (weights.max() + 1e-8) * 1.0  # Scale up: counters KL shrinkage in SVGP
        
        all_samples.append(samples)
        all_weights.append(weights)
        all_ids.append(np.full(n_samples_per_item, item_idx))
        
    samples = torch.from_numpy(np.vstack(all_samples)).float().to(device)
    weights = torch.from_numpy(np.concatenate(all_weights)).float().to(device)
    item_ids = torch.from_numpy(np.concatenate(all_ids)).long().to(device)
    
    return samples, weights, item_ids

def circular_error(pred: float, target: float) -> float:
    """Compute signed error in [-180, 180)."""
    diff = pred - target
    # Wrap to [-180, 180)
    signed = (diff + 180.0) % 360.0 - 180.0
    return signed
