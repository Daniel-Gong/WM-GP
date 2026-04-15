"""
Retrocue × Set-Size experiment: does the retrocue benefit grow with set size?

Empirical finding (e.g., Souza & Oberauer, 2016; Gunseli et al., 2015):
    The retrocue benefit (neutral MAE − cued MAE) is larger at set size 4
    than at set size 2, because more items compete for limited resources
    and selective cueing therefore confers a greater advantage.

This script runs paired neutral vs cued trials at each set size and
compares the resulting retrocue benefits.
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

SET_SIZES = [2, 4]
N_TRIALS = 100
MAINTENANCE_EPOCHS = 100


def run_retrocue_setsize_experiment():
    base_config = load_config(filename="config_retrocue.yaml")
    base_config["training"]["maintenance_epochs"] = MAINTENANCE_EPOCHS
    base_config["experiment"]["n_trials"] = N_TRIALS

    rows = []
    per_trial = {}  # set_size -> dict with trial-level arrays

    for ss in SET_SIZES:
        cfg = copy.deepcopy(base_config)

        cued_errors = []
        neutral_errors = []

        print(f"\n{'='*55}")
        print(f"  Set Size = {ss}")
        print(f"{'='*55}")

        for t in trange(N_TRIALS, desc=f"N={ss}"):
            seed = base_config["experiment"]["random_seed"] + t
            items = generator.generate_items(ss, seed=seed)
            probed_idx = np.random.RandomState(seed).randint(0, ss)

            # Neutral (no cue)
            model_n, _, _ = run_simulation_trial(items, cfg, cued_item_idx=None)
            neutral_err = abs(
                retrieve_color(model_n, items[probed_idx][0], items[probed_idx][1])
            )
            neutral_errors.append(neutral_err)

            # Cued (retrocue to probed item)
            model_c, _, _ = run_simulation_trial(items, cfg, cued_item_idx=probed_idx)
            cued_err = abs(
                retrieve_color(model_c, items[probed_idx][0], items[probed_idx][1])
            )
            cued_errors.append(cued_err)

        benefit = np.array(neutral_errors) - np.array(cued_errors)
        t_stat, p_val = stats.ttest_rel(neutral_errors, cued_errors)

        row = {
            "set_size": ss,
            "neutral_mae": np.mean(neutral_errors),
            "neutral_sem": stats.sem(neutral_errors),
            "cued_mae": np.mean(cued_errors),
            "cued_sem": stats.sem(cued_errors),
            "benefit_mean": np.mean(benefit),
            "benefit_sem": stats.sem(benefit),
            "t_stat": t_stat,
            "p_val": p_val,
        }
        rows.append(row)
        per_trial[ss] = {"neutral": neutral_errors, "cued": cued_errors, "benefit": benefit}

        print(f"  Neutral MAE  = {row['neutral_mae']:.2f} ± {row['neutral_sem']:.2f}")
        print(f"  Cued MAE     = {row['cued_mae']:.2f} ± {row['cued_sem']:.2f}")
        print(f"  Benefit      = {row['benefit_mean']:.2f} ± {row['benefit_sem']:.2f}")
        print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.4e}")

    df = pd.DataFrame(rows)

    # Compare benefits across set sizes (independent-samples t-test)
    b2 = per_trial[2]["benefit"]
    b4 = per_trial[4]["benefit"]
    t_cross, p_cross = stats.ttest_ind(b4, b2, alternative="greater")
    print(f"\n{'='*55}")
    print("  Cross-condition comparison (SS4 benefit > SS2 benefit)")
    print(f"  t = {t_cross:.3f}, p = {p_cross:.4e}")
    print(f"{'='*55}")

    df_meta = pd.DataFrame([{
        "comparison": "SS4_benefit > SS2_benefit",
        "t_stat": t_cross,
        "p_val": p_cross,
    }])

    csv_path = os.path.join(SAVE_DIR, "retrocue_setsize_results.csv")
    df.to_csv(csv_path, index=False)
    meta_path = os.path.join(SAVE_DIR, "retrocue_setsize_comparison.csv")
    df_meta.to_csv(meta_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Comparison saved to {meta_path}")

    plot_retrocue_setsize(df, p_cross)
    return df


def plot_retrocue_setsize(df: pd.DataFrame, p_cross: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Neutral vs Cued grouped bars per set size ──
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.32

    ax.bar(
        x - w / 2, df["neutral_mae"], w,
        yerr=df["neutral_sem"], capsize=4,
        label="Neutral", color="#E8834A", edgecolor="black",
    )
    ax.bar(
        x + w / 2, df["cued_mae"], w,
        yerr=df["cued_sem"], capsize=4,
        label="Cued", color="#4A90D9", edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {int(r)}" for r in df["set_size"]], fontsize=12)
    ax.set_xlabel("Set Size", fontsize=13)
    ax.set_ylabel("Mean Absolute Error (°)", fontsize=13)
    ax.set_title("Neutral vs Cued by Set Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, row in df.iterrows():
        sig = _sig_stars(row["p_val"])
        y_top = max(row["neutral_mae"] + row["neutral_sem"],
                    row["cued_mae"] + row["cued_sem"]) + 1
        ax.text(i, y_top, sig, ha="center", fontsize=12, fontweight="bold")

    # ── Panel B: Retrocue benefit comparison ──
    ax2 = axes[1]
    colors = ["#4CAF50", "#2E7D32"]
    bars = ax2.bar(
        np.arange(len(df)), df["benefit_mean"],
        yerr=df["benefit_sem"], capsize=5,
        color=colors[:len(df)], edgecolor="black", width=0.5,
    )
    ax2.set_xticks(np.arange(len(df)))
    ax2.set_xticklabels([f"N = {int(r)}" for r in df["set_size"]], fontsize=12)
    ax2.set_xlabel("Set Size", fontsize=13)
    ax2.set_ylabel("Retrocue Benefit (°)", fontsize=13)
    ax2.set_title("Retrocue Benefit: Larger at Higher Set Size?", fontsize=14)
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Significance bracket between the two bars
    y_max = max(df["benefit_mean"] + df["benefit_sem"]) + 1.5
    bracket_y = y_max + 0.5
    ax2.plot([0, 0, 1, 1], [y_max, bracket_y, bracket_y, y_max],
             color="black", linewidth=1.2)
    ax2.text(0.5, bracket_y + 0.3, _sig_stars(p_cross),
             ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "retrocue_setsize_benefit.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")


def _sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


if __name__ == "__main__":
    run_retrocue_setsize_experiment()
