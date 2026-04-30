import torch
import torch.nn.functional as F
import numpy as np
import gpytorch
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import yaml
import os
from generator import generate_items, sample_training_data, circular_error
from gp_model import WorkingMemoryGP
from attention_mechanisms import SpatialProximityAttention
import viz.visualizations as vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path=None,filename="config.yaml"):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _spatial_divisive_normalize(
    model: "WorkingMemoryGP",
    query_loc: float,
    loc_pool_width_deg: float,
    color_pool_width_deg: float,
    baseline_frac: float = 0.1,
    n_color_samples: int = 360,
    n_loc_samples: int = 36,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spatial divisive normalization (Carandini & Heeger, 2012).

    For each candidate color *c*, the normalization pool aggregates the GP
    posterior across nearby *spatial locations* (weighted by a Gaussian
    centered on ``query_loc``) and nearby *colors* (Gaussian smooth in
    color space).  This lets the distractor's peak at a different location
    contribute to the pool, creating surround suppression that shifts the
    target peak away from the distractor.

    Returns (color_samples, normalized_profile).
    """
    model.eval()

    colors = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    locs = torch.linspace(-180.0, 180.0, n_loc_samples + 1, device=device)[:-1]

    # --- evaluate the full 2-D surface in one batch ---
    L, C = torch.meshgrid(locs, colors, indexing="ij")
    grid = torch.stack([L.reshape(-1), C.reshape(-1)], dim=-1)
    with torch.no_grad():
        surface = model(grid).mean.view(n_loc_samples, n_color_samples)

    surface_shifted = surface - surface.min()

    # --- spatial weights (Gaussian centered on query_loc) ---
    loc_dists = torch.abs(locs - query_loc)
    loc_dists = torch.min(loc_dists, 360.0 - loc_dists)
    spatial_w = torch.exp(-0.5 * (loc_dists / loc_pool_width_deg) ** 2)
    spatial_w = spatial_w / spatial_w.sum()

    # Spatially-weighted pool for each color:  shape (n_color_samples,)
    spatial_pool = (surface_shifted * spatial_w.unsqueeze(1)).sum(dim=0)

    # --- optional additional Gaussian smooth along the color axis ---
    color_spacing = 360.0 / n_color_samples
    half_k = min(int(3.0 * color_pool_width_deg / color_spacing), n_color_samples // 2)
    kx = torch.arange(-half_k, half_k + 1, device=device).float() * color_spacing
    kernel = torch.exp(-0.5 * (kx / color_pool_width_deg) ** 2)
    kernel = kernel / kernel.sum()

    sp = spatial_pool.view(1, 1, -1)
    sp_padded = F.pad(sp, (half_k, half_k), mode="circular")
    pool = F.conv1d(sp_padded, kernel.view(1, 1, -1)).squeeze()

    # --- target response at query_loc ---
    query_idx = torch.argmin(torch.abs(locs - query_loc))
    target_response = surface_shifted[query_idx]

    sigma = baseline_frac * target_response.max().clamp(min=1e-8)
    normalized = target_response / (sigma + pool)

    return colors, normalized


def retrieve_color(
    model: WorkingMemoryGP,
    query_loc: float,
    true_color: float,
    n_color_samples: int = 360,
    normalization: Optional[Dict] = None,
) -> float:
    """
    Retrieve color at a given 1D location by finding the color with maximum GP prediction.

    Parameters
    ----------
    normalization : dict or None
        If provided, apply spatial divisive normalization before argmax.
        Expected keys:
            loc_pool_width  – spatial pool width (degrees)
            color_pool_width – color pool width (degrees)
            baseline_frac   – semi-saturation constant as fraction of peak
    """
    model.eval()

    if normalization is not None:
        colors, profile = _spatial_divisive_normalize(
            model,
            query_loc,
            loc_pool_width_deg=normalization["loc_pool_width"],
            color_pool_width_deg=normalization["color_pool_width"],
            baseline_frac=normalization.get("baseline_frac", 0.1),
            n_color_samples=n_color_samples,
        )
        best_idx = torch.argmax(profile)
        best_color = colors[best_idx].item()
    else:
        color_samples = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
        query_points = torch.stack([
            torch.tensor([query_loc, c], device=device) for c in color_samples
        ])
        with torch.no_grad():
            pred = model(query_points)
            means = pred.mean
            best_idx = torch.argmax(means)
            best_color = color_samples[best_idx].item()

    return circular_error(best_color, true_color)

def retrieve_color_probabilistic(
    model: WorkingMemoryGP,
    query_loc: float,
    true_color: float,
    n_color_samples: int = 360,
) -> Tuple[float, float]:
    """
    Retrieve color via posterior predictive sampling + argmax.

    Draws a single joint sample f ~ q(f) from the full GP posterior
    MultivariateNormal over all candidate colors simultaneously, then
    returns the color at which the sampled function value is highest.

    Using pred.sample() correctly respects the inter-color covariance
    induced by the GP kernel (nearby colors are positively correlated).

    WM error components emerge naturally from GP posterior geometry:
      - Narrow posterior (concentrated around true color) → low SD
      - Wide posterior (uncertain encoding)               → high SD
      - Secondary activation peak at a non-target color   → swap error
      - Flat, uninformative posterior                     → guess error

    Returns (signed_error, unsigned_error).
    """
    model.eval()
    color_samples = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    query_points = torch.stack([
        torch.tensor([query_loc, c], device=device) for c in color_samples
    ])

    with torch.no_grad():
        # Query the GP posterior: a MultivariateNormal over all query points
        # jointly, preserving the inter-color covariance from the kernel.
        pred = model(query_points)

        # Draw one joint sample: shape (n_color_samples,)
        # Adjacent colors co-vary smoothly — this is correctly captured here.
        f_sample = pred.sample()

        # Pick the color with the highest activation in this sample
        best_idx = torch.argmax(f_sample)
        recalled_color = color_samples[best_idx].item()

    return circular_error(recalled_color, true_color)

def run_simulation_trial(
    items: List[Tuple[float, float]],
    config: dict,
    encoding_epochs: int = None,
    maintenance_epochs: int = None,
    cued_item_idx: Optional[int] = None,
    track_visuals: bool = False,
    track_retrieval: bool = False,
    track_lengthscales: bool = False,
    track_loss: bool = False
) -> Tuple[WorkingMemoryGP, gpytorch.likelihoods.GaussianLikelihood, Dict]:
    """
    Runs the Encoding and Maintenance phases for a single trial of N items.
    """
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config['data']['n_samples_per_item'],
        loc_std=config['data']['loc_std'],
        color_std=config['data']['color_std'],
        loc_encoding_noise_std=config['data']['loc_encoding_noise_std'],
        color_encoding_noise_std=config['data']['color_encoding_noise_std']
    )
    
    # Model Setup
    model = WorkingMemoryGP(
        inducing_grid_size=config['model']['inducing_grid_size'],
        loc_lengthscale=config['model']['loc_lengthscale'],
        color_lengthscale=config['model']['color_lengthscale'],
        learn_inducing_locations=config['model']['learn_inducing_locations']
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']
    likelihood.noise_covar.raw_noise.requires_grad_(False)  # Fixed — not learned

    optimizer = torch.optim.Adam(
        model.parameters(),   # likelihood excluded: noise is frozen
        lr=config['training']['encoding_lr']
    )
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))
    
    history = {
        'encoding_loss': [],
        'maintenance_loss': [],
        'loc_lengthscale': [],
        'color_lengthscale': [],
        'unsigned_errors': {i: [] for i in range(len(items))},
        'signed_errors': {i: [] for i in range(len(items))}
    }
    
    # Visuals Tracking Collections (encoding + maintenance combined)
    hist_surfaces = []
    hist_ind_pts = []
    hist_ind_vals = []
    # Maintenance-only tracking for retrocue reallocation visualizations
    maint_hist_surfaces = []
    maint_hist_ind_pts  = []
    
    def log_parameters(epoch_loss, is_maint=False):

        if track_loss:
            if is_maint:
                history['maintenance_loss'].append(epoch_loss)
            else:
                history['encoding_loss'].append(epoch_loss)
            
        if track_lengthscales:
            history['loc_lengthscale'].append(model.covar_module.base_kernel.kernels[0].lengthscale.item())
            history['color_lengthscale'].append(model.covar_module.base_kernel.kernels[1].lengthscale.item())
        
        if track_retrieval:
            # Retrieval Tracking
            for i, (l, c) in enumerate(items):
                signed_err = retrieve_color(model, l, c)
                history['unsigned_errors'][i].append(abs(signed_err))
                history['signed_errors'][i].append(signed_err)
            
        if track_visuals:
            # Reconstruct 2x2 grid fast for tracking
            locs = torch.linspace(-180.0, 180.0, 100, device=device)
            cols = torch.linspace(-180.0, 180.0, 100, device=device)
            L, C = torch.meshgrid(locs, cols, indexing='ij')
            grid = torch.stack([L.flatten(), C.flatten()], dim=-1)
            
            with torch.no_grad():
                preds = likelihood(model(grid))
                hist_surfaces.append(preds.mean.view(100, 100).cpu().numpy())
                ind_pts = model.variational_strategy.inducing_points.detach()
                hist_ind_pts.append(ind_pts.cpu().numpy())
                ind_preds = likelihood(model(ind_pts)).mean.detach()
                hist_ind_vals.append(ind_preds.cpu().numpy())

    # ==================== ENCODING ====================
    model.train()
    likelihood.train()
    if encoding_epochs is None:
        encoding_epochs = config['training']['encoding_epochs']
    for epoch in range(encoding_epochs):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        
        log_parameters(loss.item(), is_maint=False)
        
    # ==================== MAINTENANCE (OPTIONAL) ====================
    if maintenance_epochs is None:
        maintenance_epochs = config['training']['maintenance_epochs']
    if maintenance_epochs > 0:
        maint_lr = config['training']['maintenance_lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = maint_lr

        # Freeze the TARGET locations (encoding-end inducing positions) but
        # allow the inducing points themselves to keep moving. This way:
        #   - Targets stay geometrically stable (no stale-target problem)
        #   - Inducing points can migrate toward the attended item during
        #     maintenance, concentrating representational precision
        #   - Attention weights are anchored to fixed spatial positions
        maint_eval_points = model.variational_strategy.inducing_points.detach().clone()
        with torch.no_grad():
            maint_targets = likelihood(model(maint_eval_points)).mean.detach()

        # Spatial attention envelope (anchored to encoding-end positions)
        if cued_item_idx is not None:
            cued_loc = items[cued_item_idx][0]
            attn = SpatialProximityAttention(
                spatial_std=config['attention']['spatial_std'],
                attended_gain=config['attention']['attended_gain']
            ).to(device)
            cued_weights = attn(maint_eval_points[:, 0], cued_loc)
        else:
            cued_weights = torch.ones(len(maint_eval_points), device=device)

        neutral_weights = torch.ones(len(maint_eval_points), device=device)

        cue_start_epoch = config['training']['cue_start_epoch']
        beta = config['training']['beta']

        for epoch in range(maintenance_epochs):
            optimizer.zero_grad()
            output = model(maint_eval_points)

            var_dist = model.variational_strategy.variational_distribution
            prior_dist = model.variational_strategy.prior_distribution
            kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)

            exp_ll = likelihood.expected_log_prob(maint_targets, output)

            if epoch >= cue_start_epoch:
                attn_weights = cued_weights
            else:
                attn_weights = neutral_weights

            weighted_ll = (exp_ll * attn_weights).sum() / len(maint_eval_points)
            loss = -weighted_ll + kl_div * beta

            loss.backward()
            optimizer.step()

            log_parameters(loss.item(), is_maint=True)

            if track_visuals:
                with torch.no_grad():
                    model.eval()
                    locs_v = torch.linspace(-180.0, 180.0, 100, device=device)
                    cols_v = torch.linspace(-180.0, 180.0, 100, device=device)
                    Lv, Cv = torch.meshgrid(locs_v, cols_v, indexing='ij')
                    grid_v = torch.stack([Lv.flatten(), Cv.flatten()], dim=-1)
                    surf_v = likelihood(model(grid_v)).mean.view(100, 100).cpu().numpy()
                    maint_hist_surfaces.append(surf_v)
                    maint_hist_ind_pts.append(
                        model.variational_strategy.inducing_points.detach().cpu().numpy()
                    )
                model.train()
            
    if track_visuals and config['output']['save_animations']:
        vis.create_gp_surface_2d_gif(hist_surfaces, hist_ind_pts, items, filename=f"gp_optimization_N={len(items)}.gif")
        vis.create_gp_surface_3d_gif(hist_surfaces, hist_ind_pts, hist_ind_vals, items, filename=f"gp_optimization_3d_N={len(items)}.gif")
    if track_loss and track_lengthscales:
        vis.plot_training_trajectories(history, filename=f"training_trajectories_N={len(items)}.png")
    if track_visuals:
        vis.plot_gp_surface_2d(model, likelihood, items, epoch="Final", prefix="", filename=f"gp_surface_N={len(items)}.png")
    if track_retrieval:
        vis.plot_item_retrieval_errors(
            history,
            items,
            cued_item_idx=cued_item_idx,
            cue_start_epoch=config['training'].get('cue_start_epoch') if cued_item_idx is not None else None,
            filename=f"item_losses_N={len(items)}.png",
        )

    # ── Retrocue allocation visualizations ──────────────────────────────────
    if track_visuals and cued_item_idx is not None and len(maint_hist_surfaces) > 0:
        cue_start = config['training']['cue_start_epoch']
        # Animated GIF across all maintenance epochs
        vis.create_retrocue_allocation_gif(
            maint_hist_surfaces,
            maint_hist_ind_pts,
            items,
            cued_item_idx=cued_item_idx,
            cue_start_epoch=cue_start,
            filename=f"retrocue_allocation_N={len(items)}.gif",
        )
        # Delta GIF: each cue-active epoch minus the last pre-cue epoch
        vis.create_retrocue_delta_gif(
            maint_hist_surfaces,
            maint_hist_ind_pts,
            items,
            cued_item_idx=cued_item_idx,
            cue_start_epoch=cue_start,
            filename=f"retrocue_delta_N={len(items)}.gif",
        )
        # Static before/after comparison (snapshot at cue onset vs final epoch)
        pre_idx  = min(cue_start - 1, len(maint_hist_surfaces) - 1)
        post_idx = len(maint_hist_surfaces) - 1
        if pre_idx >= 0 and post_idx > pre_idx:
            vis.plot_retrocue_allocation_comparison(
                surface_pre  = maint_hist_surfaces[pre_idx],
                ind_pts_pre  = maint_hist_ind_pts[pre_idx],
                surface_post = maint_hist_surfaces[post_idx],
                ind_pts_post = maint_hist_ind_pts[post_idx],
                items        = items,
                cued_item_idx= cued_item_idx,
                filename     = f"retrocue_comparison_N={len(items)}.png",
            )
        
        
    return model, likelihood, history

if __name__ == "__main__":
    # 1. Retrocue effect
    config = load_config(filename="config_retrocue.yaml")
    n_items = 4
    seed = config['experiment']['random_seed']
    demo_items = generate_items(n_items,seed=seed)
    print("Simulating a single trial with N=", n_items)
    print(demo_items)
    model, likelihood, history = run_simulation_trial(demo_items, config, cued_item_idx=0, track_visuals=True)

    # 2. Set size effect
    # n_items = 3
    # config = load_config(filename="config_set_size.yaml")
    # demo_items = generate_items(n_items)
    # print("Simulating a single trial with N=", n_items)
    # print(demo_items)
    # model, likelihood, history = run_simulation_trial(demo_items, config, maintenance_epochs=0, track_visuals=True, track_loss=True, track_lengthscales=True)
