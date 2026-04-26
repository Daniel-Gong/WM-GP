"""
3D Bias Experiment: Vary both color distance and encoding epochs.
Replicates the analysis from Chunharas et al. 2022, Experiment 4,
which crossed encoding time × color distance and found that repulsion
bias was strongest at short encoding times and small feature distances.

In our GP model, encoding_epochs is the analog of encoding time:
fewer epochs → weaker encoding → more inter-item interference → more bias.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats as sp_stats
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import run_simulation_trial, retrieve_color
from validation import load_config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_VIS_DIR = os.path.join(_REPO_ROOT, "visualizations")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_3d_bias_experiment(
    config,
    distances=(0, 20, 45, 90, 135),
    encoding_epochs_list=(50, 75, 100),
    n_trials=None,
    save_dir=None,
    nt_loc=30.0,
):
    """
    Run bias experiment varying both color distance and encoding epochs.

    Parameters
    ----------
    nt_loc : float
        Spatial location of the non-target (degrees).  Closer to the target
        (0°) increases inter-item interference via kernel overlap, matching
        the paper's same-hemifield placement (~67° angular separation with
        ~4° visual-angle gap).

    Returns
    -------
    pd.DataFrame with columns:
        Distance_deg, Encoding_Epochs, Bias_pct, Bias_deg, SEM_deg
    """
    if save_dir is None:
        save_dir = _DEFAULT_VIS_DIR
    if n_trials is None:
        n_trials = config["experiment"]["n_trials"]

    t_loc = 0.0
    normalization = config.get("normalization", None)

    total = len(distances) * len(encoding_epochs_list)
    print(
        f"\n=== 3D Bias Experiment: {len(distances)} distances "
        f"× {len(encoding_epochs_list)} encoding epochs = {total} conditions ==="
    )

    results = []

    for enc_epochs in encoding_epochs_list:
        for dist in distances:
            signed_errors = []
            desc = f"Enc={enc_epochs}, Dist={dist}°"

            for _ in trange(n_trials, desc=desc):
                t_col = int(np.random.uniform(-180.0, 180.0))
                nt_col = (t_col + dist + 180.0) % 360.0 - 180.0

                items = [(t_loc, t_col), (nt_loc, nt_col)]

                model, _, _ = run_simulation_trial(
                    items, config, encoding_epochs=enc_epochs,
                )

                s_err = retrieve_color(model, t_loc, t_col,
                                       normalization=normalization)
                signed_errors.append(s_err)

            # Distractor is at +dist → repulsion ⇒ error < 0
            num_repulsion = sum(1 for e in signed_errors if e < 0)
            bias_pct = (num_repulsion / len(signed_errors)) * 100 - 50

            mean_bias_deg = np.mean(signed_errors)
            sem_bias_deg = sp_stats.sem(signed_errors)

            results.append(
                {
                    "Distance_deg": dist,
                    "Encoding_Epochs": enc_epochs,
                    "Bias_pct": bias_pct,
                    "Bias_deg": mean_bias_deg,
                    "SEM_deg": sem_bias_deg,
                }
            )
            print(
                f"  Enc={enc_epochs}, Dist={dist}° → "
                f"Repulsion Bias: {bias_pct:.1f}%, "
                f"Mean Bias: {mean_bias_deg:.2f}° (SEM: {sem_bias_deg:.2f}°)"
            )

    df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "bias_3d_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


# ---------------------------------------------------------------------------
# 3-D bar plot (matches style of Chunharas et al. 2022, Exp 4, Fig.)
# ---------------------------------------------------------------------------

def plot_3d_bias(df, metric="Bias_pct", save_dir=None):
    """
    Create a 3D bar plot of bias vs color distance × encoding epochs.

    Parameters
    ----------
    df : pd.DataFrame  (output of run_3d_bias_experiment)
    metric : str
        Column to plot on the vertical axis.
        "Bias_pct" for % repulsion bias, "Bias_deg" for mean signed error.
    """
    if save_dir is None:
        save_dir = _DEFAULT_VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    distances = sorted(df["Distance_deg"].unique())
    enc_epochs = sorted(df["Encoding_Epochs"].unique())

    n_dist = len(distances)
    n_enc = len(enc_epochs)

    epoch_colors = ["#4A90D9", "#E8834A", "#4CAF50"]

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    bar_w = 0.55  # width  (color-distance axis)
    bar_d = 0.55  # depth  (encoding-epochs axis)

    for i, enc in enumerate(enc_epochs):
        for j, dist in enumerate(distances):
            row = df[(df["Encoding_Epochs"] == enc) & (df["Distance_deg"] == dist)]
            if row.empty:
                continue
            val = row[metric].values[0]

            x = j
            y = i
            z_bottom = min(val, 0)
            dz = abs(val)

            ax.bar3d(
                x, y, z_bottom, bar_w, bar_d, dz,
                color=epoch_colors[i % len(epoch_colors)],
                edgecolor="dimgray",
                linewidth=0.4,
                alpha=0.85,
            )

    # --- Axis ticks & labels ---
    ax.set_xticks([k + bar_w / 2 for k in range(n_dist)])
    ax.set_xticklabels([f"{d}" for d in distances], fontsize=10)
    ax.set_xlabel("color distance (°)", fontsize=12, labelpad=12)

    ax.set_yticks([k + bar_d / 2 for k in range(n_enc)])
    ax.set_yticklabels([str(e) for e in enc_epochs], fontsize=10)
    ax.set_ylabel("encoding epochs", fontsize=12, labelpad=12)

    if metric == "Bias_pct":
        zlabel = "% repulsion bias"
    else:
        zlabel = "bias (°)"
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)

    # Viewing angle mimicking the reference figure
    ax.view_init(elev=25, azim=-55)

    # Reference line at z = 0 (thin horizontal plane)
    x_range = np.array([-0.3, n_dist])
    y_range = np.array([-0.3, n_enc])
    xx, yy = np.meshgrid(x_range, y_range)
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color="gray")

    legend_handles = [
        Patch(
            facecolor=epoch_colors[i % len(epoch_colors)],
            edgecolor="dimgray",
            label=f"{enc} epochs",
        )
        for i, enc in enumerate(enc_epochs)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=10,
        framealpha=0.9,
        handlelength=1.8,
    )

    # Clean up pane colors
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    suffix = "pct" if metric == "Bias_pct" else "deg"
    filepath = os.path.join(save_dir, f"bias_3d_plot_{suffix}.png")
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"3D plot saved to {filepath}")


# ---------------------------------------------------------------------------
# 2-D line plot: one line per encoding epoch
# ---------------------------------------------------------------------------

def plot_line_bias(df, metric="Bias_deg", save_dir=None):
    """
    Overlay line plots of bias vs color distance, one line per encoding epoch.
    Error bars show ±1 SEM.
    """
    if save_dir is None:
        save_dir = _DEFAULT_VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    distances = sorted(df["Distance_deg"].unique())
    enc_epochs = sorted(df["Encoding_Epochs"].unique())

    colors = ["#4A90D9", "#E8834A", "#4CAF50","#800080"]
    markers = ["o", "s", "^","D"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, enc in enumerate(enc_epochs):
        sub = df[df["Encoding_Epochs"] == enc].sort_values("Distance_deg")
        y = sub[metric].values
        sem = sub["SEM_deg"].values if metric == "Bias_deg" else None

        ax.plot(
            sub["Distance_deg"], y,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=8,
            linewidth=2,
            label=f"{enc} epochs",
        )
        if sem is not None:
            ax.fill_between(
                sub["Distance_deg"], y - sem, y + sem,
                color=colors[i % len(colors)],
                alpha=0.18,
            )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(distances)
    ax.set_xlabel("Color Distance (°)", fontsize=13)

    if metric == "Bias_pct":
        ax.set_ylabel("% Repulsion Bias", fontsize=13)
    else:
        ax.set_ylabel("Bias (°)", fontsize=13)

    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    suffix = "pct" if metric == "Bias_pct" else "deg"
    filepath = os.path.join(save_dir, f"bias_line_plot_{suffix}.png")
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Line plot saved to {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config(filename="config_bias.yaml")

    df = run_3d_bias_experiment(
        config,
        distances=(0, 20, 45, 90, 135),
        encoding_epochs_list=(50, 75, 100, 150),
        nt_loc=30.0,
    )

    plot_3d_bias(df, metric="Bias_pct")
    plot_3d_bias(df, metric="Bias_deg")
    plot_line_bias(df, metric="Bias_deg")
    plot_line_bias(df, metric="Bias_pct")
