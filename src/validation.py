import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import generator
from simulation import run_simulation_trial, retrieve_color
import viz.visualizations as vis

def load_config(path=None, filename="config.yaml"):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", filename)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_set_size_experiment(config,save_dir="visualizations"):
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

    df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "set_size_results.csv"), index=False)
    vis.plot_error_distributions(errors_per_set_size, save_dir=save_dir)
    vis.plot_set_size_effect(df, save_dir=save_dir)
    return df

def run_retrocue_experiment(config, target_set_size=4, save_dir="visualizations"):
    """
    Validates Cued vs Neutral benefits.
    """
    from scipy import stats
    print(f"\n=== Running Retrocue Experiment (N={target_set_size}) ===")
    
    cued_errors = []
    neutral_errors = []
    
    for t in trange(config['experiment']['n_trials'], desc="Retrocue Trials"):
        seed = config['experiment']['random_seed'] + t
        items = generator.generate_items(target_set_size, seed=seed)
        
        # Determine probed index for this trial
        probed_idx = np.random.randint(0, target_set_size)
        
        # Neutral (uncued)
        model, _, _ = run_simulation_trial(items, config, cued_item_idx=None)
        uncued_err = abs(retrieve_color(model, items[probed_idx][0], items[probed_idx][1]))
        neutral_errors.append(uncued_err)
        print(f"Neutral error: {uncued_err:.4f}")
        
        # Cued
        model, _, _  = run_simulation_trial(items, config, cued_item_idx=probed_idx)
        cued_err = abs(retrieve_color(model, items[probed_idx][0], items[probed_idx][1]))
        cued_errors.append(cued_err)
        print(f"Cued error: {cued_err:.4f}")
        
    mean_neutral = np.mean(neutral_errors)
    mean_cued = np.mean(cued_errors)
    
    t_stat, p_val = stats.ttest_rel(neutral_errors, cued_errors)
    
    print(f"Neutral MAE: {mean_neutral:.4f}")
    print(f"Cued MAE: {mean_cued:.4f}")
    print(f"Retrocue Benefit: {mean_neutral - mean_cued:.4f}")
    print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4e}")
    
    os.makedirs(save_dir, exist_ok=True)
    vis.plot_retrocue_benefit(neutral_errors, cued_errors, target_set_size, save_dir=save_dir, p_val=p_val)

def run_bias_experiment(config, save_dir="visualizations"):
    """
    Validates Attraction/Repulsion Bias explicitly near non-targets using relative metric.
    We iterate over a range of color distances to see how bias strength changes.
    """
    print("\n=== Running Bias Experiment (Continuous Distance Sweep) ===")
    
    # 2 Items, varying distance in degrees
    n_items = 2
    distances = np.linspace(11.25, 90.0, 6) # From narrow to wide offsets
    
    means = []
    sems = []
    
    # Keep spatial locations fixed to isolate color-feature interference
    t_loc = 0.0
    nt_loc = 90.0
    
    # Define relative bias function locally
    def calc_bias(errs):
        num_away = sum(1 for e in errs if e < 0)
        return (num_away * 100 / len(errs)) - 50
    
    for dist in tqdm(distances, desc="Distance Sweep"):
        signed_errors = []
        
        for t in trange(config['experiment']['n_trials']):
            # Target is randomly placed on color wheel
            t_col = int(np.random.uniform(-180.0, 180.0))
            
            # Distractor is at specific distance positive from target
            nt_col = (t_col + dist + 180.0) % 360.0 - 180.0
            
            items = [(t_loc, t_col), (nt_loc, nt_col)]
                
            model, _, _ = run_simulation_trial(items, config)
            
            s_err = retrieve_color(model, t_loc, t_col)
            
            # Note: Distractor was placed in the POSITIVE direction (+dist) relative to Target
            # Therefore, if the predicted color is pulled towards the distractor, 
            # prediction > target -> signed error is positive.
            # To match the classic literature convention: Bias > 0 is Attraction, Bias < 0 is Repulsion.
            # We will use the direct signed error (in degrees) as the Trial Bias, where positive Means Attraction towards distractor.
            signed_errors.append(s_err)
            
        mean_bias_deg = np.mean(signed_errors)
        from scipy import stats as sp_stats
        sem_bias_deg = sp_stats.sem(signed_errors)
        means.append(mean_bias_deg)
        sems.append(sem_bias_deg)
        print(f"  Dist {dist:.2f} deg -> Mean Bias: {mean_bias_deg:.2f} deg (SEM: {sem_bias_deg:.2f} deg)")
        
    df = pd.DataFrame({'Distance_deg': distances, 'Bias_deg': means, 'SEM_deg': sems})
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "bias_results.csv"), index=False)
    vis.plot_bias_effect(df, save_dir=save_dir)
    return df

if __name__ == "__main__":
    
    # 2. Run Validation Suite
    # config = load_config(filename="config_set_size.yaml")
    # run_set_size_experiment(config)
    config = load_config(filename="config_retrocue.yaml")
    run_retrocue_experiment(config)
    # config = load_config(filename="config_bias.yaml")
    # run_bias_experiment(config)