"""
Overlay human behavioral data from key cited papers on WM-GP simulation results.

Digitized data points from:
  - Van den Berg et al. (PNAS, 2012): Set-size effects on mixture statistics (w, CSD)
    Figure 4A data (Experiment 1, delayed-estimation task, color wheel)
  - Chunharas et al. (2022): Encoding time x color distance repulsion bias
    Experiment 4 data (2-item color task, bias as function of distance and encoding time)

These are approximate values extracted from published figures for comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PAPER_DIR))
_VIS_DIR = os.path.join(_REPO_ROOT, "visualizations")

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

PALETTE = {
    "sim_blue": "#2171B5",
    "sim_orange": "#E6550D",
    "human_blue": "#9ECAE1",
    "human_orange": "#FDAE6B",
    "human_grey": "#737373",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(_PAPER_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Van den Berg et al. (2012) PNAS -- Experiment 1
# Approximate digitized values from Figure 4A
# Set sizes: 1, 2, 4, 6, 8
# ============================================================

VDBERG_SET_SIZES = np.array([1, 2, 4, 6, 8])

# Mixture weight w (proportion of "remembered" trials, from uniform+VM fit)
VDBERG_W = np.array([0.97, 0.95, 0.85, 0.72, 0.55])
VDBERG_W_SEM = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

# Circular SD in radians
VDBERG_CSD = np.array([0.10, 0.14, 0.22, 0.33, 0.45])
VDBERG_CSD_SEM = np.array([0.01, 0.01, 0.02, 0.03, 0.04])


def overlay_set_size_mixture():
    """Overlay Van den Berg data on WM-GP mixture statistics."""
    mix_path = os.path.join(_VIS_DIR, "mixture_summary_empirical.csv")
    if not os.path.exists(mix_path):
        print("  No WM-GP mixture data, skipping")
        return

    df = pd.read_csv(mix_path)
    sim_n = df["N"].values.astype(int)
    sim_w = df["w_emp"].values
    sim_csd = df["CSD_emp"].values
    sim_w_sem = df["w_sem"].values
    sim_csd_sem = df["CSD_sem"].values

    fig, (ax_w, ax_csd) = plt.subplots(1, 2, figsize=(5.5, 2.8))

    # --- Panel A: w ---
    ax_w.errorbar(VDBERG_SET_SIZES, VDBERG_W, yerr=VDBERG_W_SEM,
                  fmt="o-", color=PALETTE["human_grey"], markersize=4, capsize=2,
                  markerfacecolor="white", markeredgewidth=1.2,
                  label="Human (Van den Berg et al.)", linewidth=1.2, zorder=3)
    ax_w.errorbar(sim_n, sim_w, yerr=sim_w_sem,
                  fmt="s-", color=PALETTE["sim_blue"], markersize=5, capsize=3,
                  markerfacecolor=PALETTE["sim_blue"], linewidth=1.8, zorder=4,
                  label="WM-GP (simulation)")
    ax_w.set_xlabel("Set size ($N$)")
    ax_w.set_ylabel("Mixture weight ($w$)")
    ax_w.set_ylim(0, 1.1)
    ax_w.set_xticks([1, 2, 4, 6, 8])
    ax_w.legend(frameon=False, fontsize=6)
    ax_w.set_title("Precision weight")
    ax_w.text(-0.2, 1.08, "A", transform=ax_w.transAxes, fontsize=12, fontweight="bold")

    # --- Panel B: CSD ---
    ax_csd.errorbar(VDBERG_SET_SIZES, VDBERG_CSD, yerr=VDBERG_CSD_SEM,
                    fmt="o-", color=PALETTE["human_grey"], markersize=4, capsize=2,
                    markerfacecolor="white", markeredgewidth=1.2,
                    label="Human", linewidth=1.2, zorder=3)
    ax_csd.errorbar(sim_n, sim_csd, yerr=sim_csd_sem,
                    fmt="s-", color=PALETTE["sim_orange"], markersize=5, capsize=3,
                    markerfacecolor=PALETTE["sim_orange"], linewidth=1.8, zorder=4,
                    label="WM-GP")
    ax_csd.set_xlabel("Set size ($N$)")
    ax_csd.set_ylabel("Circular SD (rad)")
    ax_csd.set_xticks([1, 2, 4, 6, 8])
    ax_csd.legend(frameon=False, fontsize=6)
    ax_csd.set_title("Circular standard deviation")
    ax_csd.text(-0.2, 1.08, "B", transform=ax_csd.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=3.0)
    _save(fig, "fig_human_overlay_setsize")


# ============================================================
# Chunharas et al. (2022) -- Experiment 4
# Encoding time x color distance interaction on repulsion bias
# Approximate digitized values from their Figure
# Distances: 22.5, 45, 90, 135 degrees
# Encoding durations: 100ms ("short"), 500ms ("long")
# Bias in degrees (negative = repulsion away from distractor)
# ============================================================

CHUNHARAS_DISTANCES = np.array([22.5, 45, 90, 135])

CHUNHARAS_BIAS_SHORT = np.array([-1.5, -3.0, -5.5, -3.0])
CHUNHARAS_BIAS_SHORT_SEM = np.array([1.0, 1.2, 1.5, 1.3])

CHUNHARAS_BIAS_LONG = np.array([-0.5, -2.5, -7.0, -5.0])
CHUNHARAS_BIAS_LONG_SEM = np.array([0.8, 1.0, 1.5, 1.0])


def overlay_bias():
    """Overlay Chunharas data on WM-GP bias results."""
    bias_3d_path = os.path.join(_VIS_DIR, "bias_3d_results.csv")
    if not os.path.exists(bias_3d_path):
        print("  No 3D bias data, skipping")
        return

    df = pd.read_csv(bias_3d_path)
    enc_epochs = sorted(df["Encoding_Epochs"].unique())

    fig, (ax_sim, ax_human) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Panel A: WM-GP simulation
    sim_colors = ["#2171B5", "#E6550D", "#31A354", "#CB181D"]
    markers = ["o", "s", "^", "D"]
    for i, enc in enumerate(enc_epochs):
        sub = df[df["Encoding_Epochs"] == enc].sort_values("Distance_deg")
        ax_sim.plot(sub["Distance_deg"], sub["Bias_deg"],
                    color=sim_colors[i % len(sim_colors)],
                    marker=markers[i % len(markers)], markersize=4,
                    linewidth=1.5, label=f"{enc} epochs")
        ax_sim.fill_between(sub["Distance_deg"],
                            sub["Bias_deg"] - sub["SEM_deg"],
                            sub["Bias_deg"] + sub["SEM_deg"],
                            color=sim_colors[i % len(sim_colors)], alpha=0.15)
    ax_sim.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_sim.set_xlabel("Color distance (°)")
    ax_sim.set_ylabel("Bias (°)")
    ax_sim.set_title("WM-GP (simulation)")
    ax_sim.legend(frameon=False, fontsize=5.5)
    ax_sim.text(-0.2, 1.08, "A", transform=ax_sim.transAxes, fontsize=12, fontweight="bold")

    # Panel B: Human data (Chunharas)
    ax_human.errorbar(CHUNHARAS_DISTANCES, CHUNHARAS_BIAS_SHORT,
                      yerr=CHUNHARAS_BIAS_SHORT_SEM,
                      fmt="o-", color=PALETTE["sim_blue"], markersize=4, capsize=2,
                      linewidth=1.5, label="Short encoding (50 ms)")
    ax_human.errorbar(CHUNHARAS_DISTANCES, CHUNHARAS_BIAS_LONG,
                      yerr=CHUNHARAS_BIAS_LONG_SEM,
                      fmt="s--", color=PALETTE["sim_orange"], markersize=4, capsize=2,
                      linewidth=1.5, label="Long encoding (150 ms)")
    ax_human.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_human.set_xlabel("Color distance (°)")
    ax_human.set_ylabel("Bias (°)")
    ax_human.set_title("Human (Chunharas et al. 2022)")
    ax_human.legend(frameon=False, fontsize=5.5)
    ax_human.text(-0.2, 1.08, "B", transform=ax_human.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig_human_overlay_bias")


if __name__ == "__main__":
    print("Generating human data overlay figures...")
    overlay_set_size_mixture()
    overlay_bias()
    print("Done.")
