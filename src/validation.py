import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import generator
from simulation import run_simulation_trial, retrieve_color
import visualizations as vis

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_set_size_experiment(config):
    """
    Validates signed error distribution vs Set Size.
    Collects signed errors for each set size and plots their distributions.
    Also reports Mean Absolute Error and SD of the signed error distribution.
    """
    print("=== Running Set Size Experiment ===")
    results = []
    errors_per_set_size = {}
    
    for n_items in config['experiment']['set_sizes']:
        all_signed = []
        
        for t in trange(config['experiment']['n_trials'], desc=f"Set Size {n_items}"):
            seed = config['experiment']['random_seed'] + t
            items = generator.generate_items(n_items, seed=seed)
            model, _, _ = run_simulation_trial(items, config, cued_item_idx=None, track_visuals=False)
            
            # Read signed errors from history (final retrieval epoch per item)
            for i in range(len(items)):
                s_err = retrieve_color(model, items[i][0], items[i][1])
                all_signed.append(s_err)
                
        mean_abs_err = np.mean(np.abs(all_signed))
        sd_signed = np.std(all_signed)
        errors_per_set_size[n_items] = all_signed
        results.append({'Set Size': n_items, 'Mean Abs Error': mean_abs_err, 'SD Signed Error': sd_signed})
        print(f"Set Size {n_items} | Mean Abs Error: {mean_abs_err:.4f} | SD Signed Error: {sd_signed:.4f}")
        
    _vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(_vis_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(_vis_dir, "set_size_results.csv"), index=False)
    vis.plot_error_distributions(errors_per_set_size, save_dir=_vis_dir)
    vis.plot_set_size_effect(df, save_dir=_vis_dir)
    return df

def run_retrocue_experiment(config, target_set_size=4):
    """
    Validates Cued vs Neutral benefits.
    """
    print(f"\n=== Running Retrocue Experiment (N={target_set_size}) ===")
    
    cued_errors = []
    neutral_errors = []
    
    for t in trange(config['experiment']['n_trials'], desc="Retrocue Trials"):
        seed = config['experiment']['random_seed'] + t
        items = generator.generate_items(target_set_size, seed=seed)
        
        # Neutral (uncued)
        _, _, hist_neutral = run_simulation_trial(items, config, cued_item_idx=None)
        uncued_err = np.mean([hist_neutral['retrieval_errors'][i][-1] for i in range(target_set_size)])
        neutral_errors.append(uncued_err)
        
        # Cued
        cued_idx = np.random.randint(0, target_set_size)
        _, _, hist_cued = run_simulation_trial(items, config, cued_item_idx=cued_idx)
        cued_err = hist_cued['retrieval_errors'][cued_idx][-1]
        cued_errors.append(cued_err)
        
    print(f"Neutral MAE: {np.mean(neutral_errors):.4f}")
    print(f"Cued MAE: {np.mean(cued_errors):.4f}")
    print(f"Retrocue Benefit: {np.mean(neutral_errors) - np.mean(cued_errors):.4f}")
    _vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(_vis_dir, exist_ok=True)
    vis.plot_retrocue_benefit(np.mean(neutral_errors), np.mean(cued_errors), target_set_size, save_dir=_vis_dir)

def run_bias_experiment(config):
    """
    Validates Attraction/Repulsion Bias explicitly near non-targets using relative metric.
    We iterate over a range of color distances to see how bias strength changes.
    """
    print("\n=== Running Bias Experiment (Continuous Distance Sweep) ===")
    
    # 2 Items, varying distance in degrees
    n_items = 2
    distances = np.linspace(11.25, 90.0, 6) # From narrow to wide offsets
    
    means = []
    
    # Keep spatial locations fixed to isolate color-feature interference
    t_loc = 0.0
    nt_loc = 90.0
    
    # Define relative bias function locally
    def calc_bias(errs):
        num_away = sum(1 for e in errs if e < 0)
        return (num_away * 100 / len(errs)) - 50
    
    for dist in tqdm(distances, desc="Distance Sweep"):
        signed_errors = []
        
        for t in range(config['experiment']['n_trials']):
            # Target is randomly placed on color wheel
            t_col = np.random.uniform(-180.0, 180.0)
            
            # Distractor is at specific distance positive from target
            nt_col = (t_col + dist + 180.0) % 360.0 - 180.0
            
            items = [(t_loc, t_col), (nt_loc, nt_col)]
                
            model, likelihood, _ = run_simulation_trial(items, config)
            
            # Recompute signed error exactly at target query
            from simulation import retrieve_color
            u_err, s_err = retrieve_color(model, likelihood, t_loc, t_col)
            
            # Note: Distractor was placed in the POSITIVE direction (+dist) relative to Target
            # Therefore, if the predicted color is pulled towards the distractor, 
            # prediction > target -> signed error is positive.
            # To match the classic literature convention: Bias > 0 is Attraction, Bias < 0 is Repulsion.
            # We will use the direct signed error (in degrees) as the Trial Bias, where positive Means Attraction towards distractor.
            signed_errors.append(s_err)
            
        mean_bias_deg = np.mean(signed_errors)
        means.append(mean_bias_deg)
        print(f"  Dist {dist:.2f} deg -> Mean Bias: {mean_bias_deg:.2f} deg")
        
    df = pd.DataFrame({'Distance_deg': distances, 'Bias_deg': means})
    _vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(_vis_dir, exist_ok=True)
    df.to_csv(os.path.join(_vis_dir, "bias_results.csv"), index=False)
    vis.plot_bias_effect(df, save_dir=_vis_dir)
    return df

if __name__ == "__main__":
    config = load_config()
    
    # 1. Generate core visualizations (GIFs, 2D Plots, Trajectories) on a single high-fidelity run
    print("Generating single-trial visualizations (GP Fits, Trajectories, GIF)...")
    np.random.seed(config['experiment']['random_seed'])
    demo_items = generator.generate_items(4)
    print(demo_items)
    
    # Needs to import simulation tools correctly
    from simulation import run_simulation_trial
    # model, likelihood, history = run_simulation_trial(demo_items, config, encoding_epochs=100, maintenance_epochs=100, cued_item_idx=0, track_visuals=True)
    
    # 2. Run Validation Suite
    run_set_size_experiment(config)
    # run_retrocue_experiment(config)
    # run_bias_experiment(config)
