"""
Parameter sensitivity analysis for MemGP.

Systematically varies key parameters (inducing grid size, beta, color lengthscale)
using a small number of trials per condition to show robustness of qualitative
set-size effects and retrocue benefits.

Generates supplementary figure: fig_param_sensitivity.pdf
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PAPER_DIR))
_SRC = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC)

from simulation import run_simulation_trial, retrieve_color, load_config
from generator import generate_items

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

N_TRIALS = 5
SET_SIZES = [1, 2, 4, 6]
BASE_SEED = 42


def _run_setsize_sweep(config, label="", n_trials=N_TRIALS):
    """Run set-size experiment with given config, return dict {N: list_of_mae}."""
    results = {}
    for n_items in SET_SIZES:
        maes = []
        for t in range(n_trials):
            seed = BASE_SEED + t * 10007 + n_items
            items = generate_items(n_items, seed=seed)
            model, _, _ = run_simulation_trial(
                items, config, cued_item_idx=None, track_visuals=False
            )
            for i in range(len(items)):
                err = abs(retrieve_color(model, items[i][0], items[i][1]))
                maes.append(err)
        results[n_items] = maes
    return results


def param_sensitivity_inducing_grid():
    """Vary inducing_grid_size: 6, 10, 16."""
    config_base = load_config(filename="config_set_size.yaml")
    config_base["experiment"]["n_trials"] = N_TRIALS
    config_base["training"]["maintenance_epochs"] = 0

    grid_sizes = [6, 10, 16]
    all_results = {}

    for gs in grid_sizes:
        cfg = copy.deepcopy(config_base)
        cfg["model"]["inducing_grid_size"] = gs
        print(f"  Inducing grid = {gs}x{gs} = {gs**2} points ...")
        all_results[gs] = _run_setsize_sweep(cfg, f"grid={gs}")

    return all_results, grid_sizes, "Inducing grid size ($M^{1/2}$)"


def param_sensitivity_beta():
    """Vary beta (KL weight): 0.1, 1, 5."""
    config_base = load_config(filename="config_set_size.yaml")
    config_base["experiment"]["n_trials"] = N_TRIALS
    config_base["training"]["maintenance_epochs"] = 50

    betas = [0.1, 1.0, 5.0]
    all_results = {}

    for beta in betas:
        cfg = copy.deepcopy(config_base)
        cfg["training"]["beta"] = beta
        print(f"  beta = {beta} ...")
        all_results[beta] = _run_setsize_sweep(cfg, f"beta={beta}")

    return all_results, betas, "KL weight ($\\beta$)"


def param_sensitivity_lengthscale():
    """Vary color_lengthscale: 5, 10, 20, 40."""
    config_base = load_config(filename="config_set_size.yaml")
    config_base["experiment"]["n_trials"] = N_TRIALS
    config_base["training"]["maintenance_epochs"] = 0

    lengthscales = [5.0, 10.0, 20.0]
    all_results = {}

    for ls in lengthscales:
        cfg = copy.deepcopy(config_base)
        cfg["model"]["color_lengthscale"] = ls
        cfg["model"]["loc_lengthscale"] = ls
        print(f"  lengthscale = {ls} ...")
        all_results[ls] = _run_setsize_sweep(cfg, f"ls={ls}")

    return all_results, lengthscales, "Kernel lengthscale ($\\lambda$)"


def plot_sensitivity(all_experiments, save_name="fig_param_sensitivity"):
    """Plot 1x3 panel of parameter sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    panel_labels = "ABC"
    colors_cycle = ["#2171B5", "#E6550D", "#31A354", "#CB181D"]

    for ax_idx, (results, param_vals, xlabel) in enumerate(all_experiments):
        ax = axes[ax_idx]
        for i, pv in enumerate(param_vals):
            data = results[pv]
            means = []
            sems = []
            for n in SET_SIZES:
                errs = np.array(data[n])
                means.append(np.mean(errs))
                sems.append(np.std(errs) / np.sqrt(len(errs)))
            ax.errorbar(
                SET_SIZES, means, yerr=sems,
                fmt="o-", color=colors_cycle[i % len(colors_cycle)],
                markersize=4, capsize=2, linewidth=1.5,
                label=f"{pv}"
            )
        ax.set_xlabel("Set size ($N$)")
        ax.set_ylabel("MAE (°)")
        ax.set_title(xlabel)
        ax.set_xticks(SET_SIZES)
        ax.legend(frameon=False, fontsize=5.5)
        ax.text(-0.2, 1.08, panel_labels[ax_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_PAPER_DIR, f"{save_name}.{ext}"))
    plt.close(fig)
    print(f"  Saved {save_name}")


if __name__ == "__main__":
    print("Running parameter sensitivity analysis...")
    print("\n[1/3] Inducing grid size")
    r1 = param_sensitivity_inducing_grid()
    print("\n[2/3] Beta (KL weight)")
    r2 = param_sensitivity_beta()
    print("\n[3/3] Kernel lengthscale")
    r3 = param_sensitivity_lengthscale()

    plot_sensitivity([r1, r2, r3])
    print("\nDone.")
