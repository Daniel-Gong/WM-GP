"""
Cue-timing experiment: does earlier cueing produce a larger retrocue benefit?

Fixed parameters:
    - maintenance_epochs = 200
    - set_size = 4
    - n_trials = 50

Manipulated:
    - cue_start_epoch ∈ {50, 150}

For each cue_start_epoch we run a paired neutral vs cued comparison and
report the retrocue benefit (neutral MAE − cued MAE).
"""

import os
import copy
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import trange
import matplotlib.pyplot as plt

import generator
from simulation import load_config, run_simulation_trial, retrieve_color

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "visualizations")
os.makedirs(SAVE_DIR, exist_ok=True)

CUE_START_EPOCHS = [50, 150]
N_TRIALS = 50
SET_SIZE = 4
MAINTENANCE_EPOCHS = 200


def run_cue_timing_experiment():
    base_config = load_config(filename="config_retrocue.yaml")
    base_config["training"]["maintenance_epochs"] = MAINTENANCE_EPOCHS
    base_config["experiment"]["n_trials"] = N_TRIALS

    rows = []

    for cse in CUE_START_EPOCHS:
        cfg = copy.deepcopy(base_config)
        cfg["training"]["cue_start_epoch"] = cse

        cued_errors = []
        neutral_errors = []

        print(f"\n{'='*50}")
        print(f"  cue_start_epoch = {cse}  (cue active for {MAINTENANCE_EPOCHS - cse} epochs)")
        print(f"{'='*50}")

        for t in trange(N_TRIALS, desc=f"cue_start={cse}"):
            seed = base_config["experiment"]["random_seed"] + t
            items = generator.generate_items(SET_SIZE, seed=seed)
            probed_idx = np.random.RandomState(seed).randint(0, SET_SIZE)

            # Neutral
            model_n, _, _ = run_simulation_trial(items, cfg, cued_item_idx=None)
            neutral_err = abs(retrieve_color(model_n, items[probed_idx][0], items[probed_idx][1]))
            neutral_errors.append(neutral_err)

            # Cued
            model_c, _, _ = run_simulation_trial(items, cfg, cued_item_idx=probed_idx)
            cued_err = abs(retrieve_color(model_c, items[probed_idx][0], items[probed_idx][1]))
            cued_errors.append(cued_err)

        benefit = np.array(neutral_errors) - np.array(cued_errors)
        t_stat, p_val = stats.ttest_rel(neutral_errors, cued_errors)

        row = {
            "cue_start_epoch": cse,
            "cue_duration": MAINTENANCE_EPOCHS - cse,
            "neutral_mae": np.mean(neutral_errors),
            "cued_mae": np.mean(cued_errors),
            "benefit_mean": np.mean(benefit),
            "benefit_sem": stats.sem(benefit),
            "t_stat": t_stat,
            "p_val": p_val,
        }
        rows.append(row)
        print(f"  Neutral MAE  = {row['neutral_mae']:.2f}")
        print(f"  Cued MAE     = {row['cued_mae']:.2f}")
        print(f"  Benefit      = {row['benefit_mean']:.2f} ± {row['benefit_sem']:.2f}")
        print(f"  t = {t_stat:.3f}, p = {p_val:.4e}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "cue_timing_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    plot_cue_timing(df)
    return df


def plot_cue_timing(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: grouped bar chart (Neutral vs Cued per condition) ---
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    bars_n = ax.bar(x - w / 2, df["neutral_mae"], w,
                    yerr=df["benefit_sem"], capsize=4,
                    label="Neutral", color="lightcoral", edgecolor="black")
    bars_c = ax.bar(x + w / 2, df["cued_mae"], w,
                    capsize=4,
                    label="Cued", color="lightgreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {int(r)}" for r in df["cue_start_epoch"]])
    ax.set_xlabel("Cue onset", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (deg)", fontsize=12)
    ax.set_title("Neutral vs Cued by Cue Onset", fontsize=13)
    ax.legend()

    for i, row in df.iterrows():
        sig = "***" if row.p_val < .001 else ("**" if row.p_val < .01 else ("*" if row.p_val < .05 else "ns"))
        y_top = max(row.neutral_mae, row.cued_mae) + row.benefit_sem + 1
        ax.text(i, y_top, sig, ha="center", fontsize=11)

    # --- Panel B: benefit as a function of cue onset ---
    ax2 = axes[1]
    ax2.errorbar(df["cue_start_epoch"], df["benefit_mean"],
                 yerr=df["benefit_sem"], fmt="o-", color="steelblue",
                 capsize=5, linewidth=2, markersize=8)
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Cue onset epoch", fontsize=12)
    ax2.set_ylabel("Retrocue Benefit (deg)", fontsize=12)
    ax2.set_title("Earlier Cue → Larger Benefit?", fontsize=13)
    ax2.set_xticks(df["cue_start_epoch"].tolist())

    for _, row in df.iterrows():
        sig = "***" if row.p_val < .001 else ("**" if row.p_val < .01 else ("*" if row.p_val < .05 else "ns"))
        ax2.annotate(sig,
                     xy=(row.cue_start_epoch, row.benefit_mean + row.benefit_sem + 0.3),
                     ha="center", fontsize=11)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "cue_timing_benefit.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    run_cue_timing_experiment()
