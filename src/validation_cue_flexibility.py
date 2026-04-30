"""
Cue-flexibility experiment: can the retrocue flexibly shift its benefit
from one item to another within a single trial?

Design
------
- set_size = 4, maintenance_epochs = 150, encoding_epochs = 100
- Epoch   0– 49: neutral (no cue)
- Epoch  50– 99: cue item A  (first cue)
- Epoch 100–149: cue item B  (second cue)

Measurements (per trial)
    ① At epoch 100 (end of 1st-cue phase): accuracy of items A, B, and others
    ② At epoch 150 (end of 2nd-cue phase): accuracy of items A, B, and others

Prediction: if the model is flexible, item A should be best at epoch 100,
but item B should catch up (or surpass) by epoch 150, demonstrating a
dynamic reallocation of the retrocue benefit.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import trange
import torch
import gpytorch
import matplotlib.pyplot as plt

import generator
from simulation import (
    load_config,
    retrieve_color,
    device,
)
from generator import sample_training_data
from gp_model import WorkingMemoryGP
from attention_mechanisms import SpatialProximityAttention

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "visualizations")
os.makedirs(SAVE_DIR, exist_ok=True)

N_TRIALS = 50
SET_SIZE = 4
MAINTENANCE_EPOCHS = 150
ENCODING_EPOCHS = 100
CUE_A_START = 50
CUE_B_START = 100


def run_flexibility_trial(
    items,
    config,
    cued_idx_A: int,
    cued_idx_B: int,
):
    """
    Run a single trial with two sequential retrocues.

    Returns (snapshot_post_A, snapshot_post_B) where each is a dict
    mapping item_index -> absolute retrieval error at that timepoint.
    """
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config["data"]["n_samples_per_item"],
        loc_std=config["data"]["loc_std"],
        color_std=config["data"]["color_std"],
        loc_encoding_noise_std=config["data"]["loc_encoding_noise_std"],
        color_encoding_noise_std=config["data"]["color_encoding_noise_std"],
    )

    model = WorkingMemoryGP(
        inducing_grid_size=config["model"]["inducing_grid_size"],
        loc_lengthscale=config["model"]["loc_lengthscale"],
        color_lengthscale=config["model"]["color_lengthscale"],
        learn_inducing_locations=config["model"]["learn_inducing_locations"],
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config["likelihood"]["noise_variance"]
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["encoding_lr"])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))

    # ── Encoding ──
    model.train()
    likelihood.train()
    for _ in range(ENCODING_EPOCHS):
        optimizer.zero_grad()
        loss = -mll(model(samples), weights)
        loss.backward()
        optimizer.step()

    # ── Maintenance with dual sequential cues ──
    maint_lr = config["training"]["maintenance_lr"]
    for pg in optimizer.param_groups:
        pg["lr"] = maint_lr

    # Frozen evaluation points & targets from encoding end; inducing points
    # remain learnable so they can migrate during maintenance.
    maint_eval_points = model.variational_strategy.inducing_points.detach().clone()
    with torch.no_grad():
        maint_targets = likelihood(model(maint_eval_points)).mean.detach()

    attn_module = SpatialProximityAttention(
        spatial_std=config["attention"]["spatial_std"],
        attended_gain=config["attention"]["attended_gain"],
    ).to(device)

    cue_A_weights = attn_module(maint_eval_points[:, 0], items[cued_idx_A][0])
    cue_B_weights = attn_module(maint_eval_points[:, 0], items[cued_idx_B][0])
    neutral_weights = torch.ones(len(maint_eval_points), device=device)

    beta = config["training"]["beta"]
    snapshot_post_A = None

    for epoch in range(MAINTENANCE_EPOCHS):
        optimizer.zero_grad()
        output = model(maint_eval_points)

        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
        exp_ll = likelihood.expected_log_prob(maint_targets, output)

        if epoch < CUE_A_START:
            attn_w = neutral_weights
        elif epoch < CUE_B_START:
            attn_w = cue_A_weights
        else:
            attn_w = cue_B_weights

        weighted_ll = (exp_ll * attn_w).sum() / len(maint_eval_points)
        loss = -weighted_ll + kl_div * beta
        loss.backward()
        optimizer.step()

        # Snapshot at end of cue-A phase (just before cue-B starts)
        if epoch == CUE_B_START - 1:
            model.eval()
            snapshot_post_A = {}
            for i, (loc, col) in enumerate(items):
                snapshot_post_A[i] = abs(retrieve_color(model, loc, col))
            model.train()

    # Snapshot at end of cue-B phase (final epoch)
    model.eval()
    snapshot_post_B = {}
    for i, (loc, col) in enumerate(items):
        snapshot_post_B[i] = abs(retrieve_color(model, loc, col))

    return snapshot_post_A, snapshot_post_B


def run_flexibility_experiment():
    base_config = load_config(filename="config_retrocue.yaml")
    base_config["training"]["maintenance_epochs"] = MAINTENANCE_EPOCHS
    base_config["training"]["encoding_epochs"] = ENCODING_EPOCHS
    base_config["training"]["cue_start_epoch"] = CUE_A_START

    rows = []

    print(f"{'='*60}")
    print(f"  Cue-Flexibility Experiment")
    print(f"  Set size={SET_SIZE}, maint epochs={MAINTENANCE_EPOCHS}")
    print(f"  Cue A: epochs {CUE_A_START}–{CUE_B_START-1}")
    print(f"  Cue B: epochs {CUE_B_START}–{MAINTENANCE_EPOCHS-1}")
    print(f"  N trials = {N_TRIALS}")
    print(f"{'='*60}")

    for t in trange(N_TRIALS, desc="Flexibility trials"):
        seed = base_config["experiment"]["random_seed"] + t
        items = generator.generate_items(SET_SIZE, seed=seed)

        rng = np.random.RandomState(seed)
        idx_A, idx_B = rng.choice(SET_SIZE, size=2, replace=False)

        snap_A, snap_B = run_flexibility_trial(
            items, base_config, cued_idx_A=idx_A, cued_idx_B=idx_B,
        )

        other_idxs = [i for i in range(SET_SIZE) if i not in (idx_A, idx_B)]

        rows.append({
            "trial": t,
            "A_post_cueA": snap_A[idx_A],
            "B_post_cueA": snap_A[idx_B],
            "other_post_cueA": np.mean([snap_A[i] for i in other_idxs]),
            "A_post_cueB": snap_B[idx_A],
            "B_post_cueB": snap_B[idx_B],
            "other_post_cueB": np.mean([snap_B[i] for i in other_idxs]),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "cue_flexibility_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    _print_stats(df)
    plot_flexibility(df)
    return df


def _print_stats(df):
    print(f"\n{'─'*50}")
    print("  Mean Absolute Error (degrees)")
    print(f"{'─'*50}")
    print(f"  {'':20s} {'Post Cue-A':>12s}  {'Post Cue-B':>12s}")
    print(f"  {'Item A (1st cued)':20s} {df['A_post_cueA'].mean():12.2f}  {df['A_post_cueB'].mean():12.2f}")
    print(f"  {'Item B (2nd cued)':20s} {df['B_post_cueA'].mean():12.2f}  {df['B_post_cueB'].mean():12.2f}")
    print(f"  {'Other items':20s} {df['other_post_cueA'].mean():12.2f}  {df['other_post_cueB'].mean():12.2f}")

    # Key contrast: does B improve from post-A to post-B?
    t1, p1 = stats.ttest_rel(df["B_post_cueA"], df["B_post_cueB"])
    print(f"\n  Item B improvement (post-A → post-B): "
          f"Δ = {df['B_post_cueA'].mean() - df['B_post_cueB'].mean():.2f}°, "
          f"t = {t1:.3f}, p = {p1:.4e}")

    # Does A degrade after cue shifts to B?
    t2, p2 = stats.ttest_rel(df["A_post_cueA"], df["A_post_cueB"])
    print(f"  Item A change      (post-A → post-B): "
          f"Δ = {df['A_post_cueA'].mean() - df['A_post_cueB'].mean():.2f}°, "
          f"t = {t2:.3f}, p = {p2:.4e}")


def plot_flexibility(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    timepoints = ["Post Cue-A\n(epoch 100)", "Post Cue-B\n(epoch 150)"]
    x = np.array([0, 1])

    # ── Panel A: paired line plot for each item role ──
    ax = axes[0]
    roles = {
        "Item A (1st cued)": ("A_post_cueA", "A_post_cueB", "#E63946"),
        "Item B (2nd cued)": ("B_post_cueA", "B_post_cueB", "#457B9D"),
        "Other items":       ("other_post_cueA", "other_post_cueB", "#999999"),
    }
    for label, (col_t1, col_t2, color) in roles.items():
        means = [df[col_t1].mean(), df[col_t2].mean()]
        sems = [stats.sem(df[col_t1]), stats.sem(df[col_t2])]
        ax.errorbar(x, means, yerr=sems, fmt="o-", color=color,
                    label=label, linewidth=2, markersize=8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_ylabel("Mean Absolute Error (°)", fontsize=12)
    ax.set_title("Cue Flexibility: Error by Timepoint", fontsize=13)
    ax.legend(fontsize=10)

    # ── Panel B: bar chart at the two timepoints ──
    ax2 = axes[1]
    w = 0.25
    labels_short = ["A", "B", "Other"]
    cols_t1 = ["A_post_cueA", "B_post_cueA", "other_post_cueA"]
    cols_t2 = ["A_post_cueB", "B_post_cueB", "other_post_cueB"]
    bar_colors = ["#E63946", "#457B9D", "#999999"]

    x_bar = np.arange(len(labels_short))
    for i, (c1, c2, bc) in enumerate(zip(cols_t1, cols_t2, bar_colors)):
        ax2.bar(x_bar[i] - w/2, df[c1].mean(), w,
                yerr=stats.sem(df[c1]), capsize=4,
                color=bc, alpha=0.5, edgecolor="black", label="Post Cue-A" if i == 0 else "")
        ax2.bar(x_bar[i] + w/2, df[c2].mean(), w,
                yerr=stats.sem(df[c2]), capsize=4,
                color=bc, edgecolor="black", label="Post Cue-B" if i == 0 else "")

    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(labels_short, fontsize=11)
    ax2.set_ylabel("Mean Absolute Error (°)", fontsize=12)
    ax2.set_title("Error by Item Role & Timepoint", fontsize=13)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.5, edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=1.0, edgecolor="black"),
    ]
    ax2.legend(handles, ["Post Cue-A", "Post Cue-B"], fontsize=10)

    # ── Panel C: Δ error (shift in benefit) ──
    ax3 = axes[2]
    delta_A = df["A_post_cueA"] - df["A_post_cueB"]
    delta_B = df["B_post_cueA"] - df["B_post_cueB"]
    delta_other = df["other_post_cueA"] - df["other_post_cueB"]

    deltas = [delta_A, delta_B, delta_other]
    d_means = [d.mean() for d in deltas]
    d_sems = [stats.sem(d) for d in deltas]
    bar_x = np.arange(3)

    bars = ax3.bar(bar_x, d_means, yerr=d_sems, capsize=5,
                   color=bar_colors, edgecolor="black")
    ax3.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax3.set_xticks(bar_x)
    ax3.set_xticklabels(["Item A\n(1st cued)", "Item B\n(2nd cued)", "Other"], fontsize=11)
    ax3.set_ylabel("Δ Error (post-A − post-B) (°)", fontsize=12)
    ax3.set_title("Benefit Shift: + = got worse, − = improved", fontsize=13)

    for i, (dm, ds, delta) in enumerate(zip(d_means, d_sems, deltas)):
        t_stat, p_val = stats.ttest_1samp(delta, 0)
        sig = "***" if p_val < .001 else ("**" if p_val < .01 else ("*" if p_val < .05 else "ns"))
        y_pos = dm + ds + 0.3 if dm >= 0 else dm - ds - 0.8
        ax3.text(i, y_pos, sig, ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, "cue_flexibility.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    run_flexibility_experiment()
