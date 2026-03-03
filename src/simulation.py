import torch
import numpy as np
import gpytorch
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

from generator import sample_training_data, circular_error
from gp_model import WorkingMemoryGP
from attention_mechanisms import SpatialProximityAttention
# Ensure visualizations logic is loaded properly
import visualizations as vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def retrieve_color(
    model: WorkingMemoryGP,
    query_loc: float,
    true_color: float,
    n_color_samples: int = 360
) -> Tuple[float, float]:
    """
    Retrieve color at a given 1D location by finding the color with maximum GP prediction.
    Both queries and truths span [-pi, pi)
    Returns (unsigned_error, signed_error).
    """
    model.eval()
    
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
    cued_item_idx: Optional[int] = None,
    track_visuals: bool = False
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
    
    # Visuals Tracking Collections
    hist_surfaces = []
    hist_ind_pts = []
    hist_ind_vals = []
    
    def log_parameters(epoch_loss, is_maint=False):
        if is_maint:
            history['maintenance_loss'].append(epoch_loss)
        else:
            history['encoding_loss'].append(epoch_loss)
            
        history['loc_lengthscale'].append(model.covar_module.base_kernel.kernels[0].lengthscale.item())
        history['color_lengthscale'].append(model.covar_module.base_kernel.kernels[1].lengthscale.item())
        
        # Retrieval Tracking
        for i, (l, c) in enumerate(items):
            signed_err = retrieve_color(model, l, c)
            history['unsigned_errors'][i].append(abs(signed_err))
            history['signed_errors'][i].append(signed_err)
            
        if track_visuals:
            # Reconstruct 2x2 grid fast for tracking
            locs = torch.linspace(-180.0, 180.0, 50, device=device)
            cols = torch.linspace(-180.0, 180.0, 50, device=device)
            L, C = torch.meshgrid(locs, cols, indexing='ij')
            grid = torch.stack([L.flatten(), C.flatten()], dim=-1)
            
            with torch.no_grad():
                preds = likelihood(model(grid))
                hist_surfaces.append(preds.mean.view(50, 50).cpu().numpy())
                ind_pts = model.variational_strategy.inducing_points.detach()
                hist_ind_pts.append(ind_pts.cpu().numpy())
                ind_preds = likelihood(model(ind_pts)).mean.detach()
                hist_ind_vals.append(ind_preds.cpu().numpy())

    # ==================== ENCODING ====================
    model.train()
    likelihood.train()
    
    for epoch in range(config['training']['encoding_epochs']):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        
        if track_visuals:
            log_parameters(loss.item(), is_maint=False)
        
    # ==================== MAINTENANCE (OPTIONAL) ====================
    if config['training']['maintenance_epochs'] > 0:
        # Switch to maintenance learning rate (typically smaller than encoding lr
        # to avoid disrupting the learned representation during self-rehearsal)
        maint_lr = config['training']['maintenance_lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = maint_lr

        # Use inducing points directly as the maintenance rehearsal set
        maint_grid = model.variational_strategy.inducing_points.detach()
        
        with torch.no_grad():
            maint_weights = likelihood(model(maint_grid)).mean.detach()
            
        # Determine strict 1D spatial attention envelope
        if cued_item_idx is not None:
            cued_loc = items[cued_item_idx][0]
            attn = SpatialProximityAttention(
                spatial_std=config['attention']['spatial_std'],
                attended_gain=config['attention']['attended_gain']
            ).to(device)
            cued_weights = attn(maint_grid[:, 0], cued_loc)
        else:
            cued_weights = torch.ones(len(maint_grid), device=device)

        neutral_weights = torch.ones(len(maint_grid), device=device)
        
        # Get timing parameters
        cue_start_epoch = config['training']['cue_start_epoch']

        for epoch in range(config['training']['maintenance_epochs']):
            optimizer.zero_grad()
            output = model(maint_grid)
            
            var_dist = model.variational_strategy.variational_distribution
            prior_dist = model.variational_strategy.prior_distribution
            kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
            
            # Calculate expected log-likelihood
            exp_ll = likelihood.expected_log_prob(maint_weights, output)

            # Determine which attention weights to use
            if epoch >= cue_start_epoch:
                attn_weights = cued_weights
            else:
                attn_weights = neutral_weights

            weighted_ll = (exp_ll * attn_weights).sum() / len(maint_grid)
            loss = -weighted_ll + kl_div
            
            loss.backward()
            optimizer.step()
            
            if track_visuals:
                log_parameters(loss.item(), is_maint=True)
            
    if track_visuals and config['output']['save_animations']:
        vis.create_gp_surface_2d_gif(hist_surfaces, hist_ind_pts, items, filename=f"gp_optimization_N{len(items)}.gif")
        vis.create_gp_surface_3d_gif(hist_surfaces, hist_ind_pts, hist_ind_vals, items, filename=f"gp_optimization_3d_N{len(items)}.gif")
        
    return model, likelihood, history
