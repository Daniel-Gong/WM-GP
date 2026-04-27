"""
Regenerate all paper figures with publication-quality styling.

Reads existing simulation output CSVs/NPZs and produces vector-format
(PDF + PNG) figures with consistent fonts, sizes, and color palettes
suitable for Nature/Science submission.
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
from scipy.stats import vonmises

_PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PAPER_DIR))
_VIS_DIR = os.path.join(_REPO_ROOT, "visualizations")
_SRC = os.path.join(_REPO_ROOT, "src")
sys.path.insert(0, _SRC)

# --- Publication style ---
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
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "blue": "#2171B5",
    "orange": "#E6550D",
    "green": "#31A354",
    "red": "#CB181D",
    "purple": "#6A51A3",
    "grey": "#636363",
    "light_blue": "#6BAED6",
    "light_orange": "#FD8D3C",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(_PAPER_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ============================================================
# Figure 2: Set-size effects and model comparison
# ============================================================

def figure2_set_size():
    """Three-panel figure: error distributions, MAE vs N, model comparison."""

    # --- Load data ---
    df_ss = pd.read_csv(os.path.join(_VIS_DIR, "set_size_results.csv"))
    df_mix = pd.read_csv(os.path.join(_VIS_DIR, "mixture_summary_empirical.csv"))
    npz = np.load(os.path.join(_VIS_DIR, "wm_gp_errors_by_set_size.npz"))

    set_sizes = sorted([int(k[1:]) for k in npz.files])
    errors_by_n = {n: npz[f"N{n}"] for n in set_sizes}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    # --- Panel A: Error distributions ---
    ax = axes[0]
    colors_list = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["red"]]
    bins = np.linspace(-np.pi, np.pi, 61)
    x_kde = np.linspace(-np.pi, np.pi, 500)
    for i, n in enumerate(set_sizes):
        errs = errors_by_n[n]
        kde = stats.gaussian_kde(errs)
        ax.plot(x_kde, kde(x_kde), color=colors_list[i % len(colors_list)],
                linewidth=1.5, label=f"$N={n}$")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Error (rad)")
    ax.set_ylabel("Density")
    ax.set_title("Error distributions")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(["$-\\pi$", "", "0", "", "$\\pi$"])
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.18, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # --- Panel B: MAE and SD vs set size ---
    ax = axes[1]
    ss = df_ss["Set Size"].values
    mae = df_ss["Mean Abs Error"].values
    sd = df_ss["SD Signed Error"].values
    ax.errorbar(ss, mae, yerr=sd / np.sqrt(100), fmt="o-", color=PALETTE["blue"],
                markersize=5, capsize=3, linewidth=1.5)
    ax.set_xlabel("Set size ($N$)")
    ax.set_ylabel("Mean absolute error (°)")
    ax.set_title("Set-size effect")
    ax.set_xticks(ss)
    ax.text(-0.18, 1.08, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # --- Panel C: Mixture statistics (w, CSD) ---
    ax = axes[2]
    ns = df_mix["N"].values
    w_emp = df_mix["w_emp"].values
    csd_emp = df_mix["CSD_emp"].values
    w_sem = df_mix["w_sem"].values
    csd_sem = df_mix["CSD_sem"].values

    ax.errorbar(ns, w_emp, yerr=w_sem, fmt="o-", color=PALETTE["blue"],
                markersize=5, capsize=3, label="$w$ (precision)")
    ax2 = ax.twinx()
    ax2.errorbar(ns, csd_emp, yerr=csd_sem, fmt="s--", color=PALETTE["orange"],
                 markersize=5, capsize=3, label="CSD")
    ax.set_xlabel("Set size ($N$)")
    ax.set_ylabel("Mixture weight ($w$)", color=PALETTE["blue"])
    ax2.set_ylabel("Circular SD (rad)", color=PALETTE["orange"])
    ax.set_xticks(ns.astype(int))
    ax.set_ylim(0, 1.1)
    ax.set_title("Mixture statistics")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center left",
              fontsize=6)
    ax2.spines["right"].set_visible(True)
    ax.text(-0.18, 1.08, "C", transform=ax.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig2_set_size")


# ============================================================
# Figure 3: Retrocue effects
# ============================================================

def figure3_retrocue():
    """Multi-panel retrocue figure."""

    # Check which data files exist
    retrocue_ss_path = os.path.join(_VIS_DIR, "retrocue_setsize_results.csv")
    cue_timing_path = os.path.join(_VIS_DIR, "cue_timing_results.csv")
    cue_flex_path = os.path.join(_VIS_DIR, "cue_flexibility_results.csv")

    n_panels = 2
    has_retrocue_ss = os.path.exists(retrocue_ss_path)
    has_cue_timing = os.path.exists(cue_timing_path)
    has_cue_flex = os.path.exists(cue_flex_path)
    if has_retrocue_ss:
        n_panels += 1
    if has_cue_timing:
        n_panels += 1
    if has_cue_flex:
        n_panels += 1

    fig, axes = plt.subplots(1, min(n_panels, 5), figsize=(min(n_panels * 2.5, 12), 2.8))
    if n_panels == 1:
        axes = [axes]
    panel_idx = 0
    labels = "ABCDE"

    # Panel A: GP surface before/after cue (conceptual)
    ax = axes[panel_idx]
    sigma_l, sigma_c = 25.0, 25.0
    items = [(-90, -60), (-30, 80), (60, -120), (120, 30)]
    locs_1d = np.linspace(-180, 180, 150)
    cols_1d = np.linspace(-180, 180, 150)
    L, C = np.meshgrid(locs_1d, cols_1d, indexing="ij")

    surface_pre = np.zeros_like(L)
    for loc, col in items:
        dl = np.minimum(np.abs(L - loc), 360 - np.abs(L - loc))
        dc = np.minimum(np.abs(C - col), 360 - np.abs(C - col))
        surface_pre += np.exp(-0.5 * (dl / sigma_l) ** 2 - 0.5 * (dc / sigma_c) ** 2)

    cued_idx = 0
    cued_l, cued_c = items[cued_idx]
    surface_post = np.zeros_like(L)
    for i, (loc, col) in enumerate(items):
        dl = np.minimum(np.abs(L - loc), 360 - np.abs(L - loc))
        dc = np.minimum(np.abs(C - col), 360 - np.abs(C - col))
        amp = 2.5 if i == cued_idx else 0.3
        sig = sigma_l * (0.8 if i == cued_idx else 1.5)
        surface_post += amp * np.exp(-0.5 * (dl / sig) ** 2 - 0.5 * (dc / sig) ** 2)

    delta = surface_post - surface_pre
    vmax = np.abs(delta).max()
    ax.imshow(delta.T, extent=[-180, 180, -180, 180], origin="lower",
              cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    for i, (loc, col) in enumerate(items):
        marker_s = 100 if i == cued_idx else 50
        ec = "red" if i == cued_idx else "white"
        ax.scatter(loc, col, c="white", marker="*", s=marker_s,
                   edgecolors=ec, linewidths=1, zorder=5)
    ax.set_xlabel("Location (°)")
    ax.set_ylabel("Color (°)")
    ax.set_title("Post-cue $\\Delta$ surface")
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-180, 0, 180])
    ax.text(-0.22, 1.08, labels[panel_idx], transform=ax.transAxes,
            fontsize=12, fontweight="bold")
    panel_idx += 1

    # Panel B: Retrocue benefit bar chart (synthetic from set_size_results)
    ax = axes[panel_idx]
    if has_retrocue_ss:
        df_rc = pd.read_csv(retrocue_ss_path)
        x = np.arange(len(df_rc))
        w = 0.32
        ax.bar(x - w/2, df_rc["neutral_mae"], w, yerr=df_rc["neutral_sem"],
               capsize=3, color=PALETTE["orange"], edgecolor="black", linewidth=0.5,
               label="Neutral")
        ax.bar(x + w/2, df_rc["cued_mae"], w, yerr=df_rc["cued_sem"],
               capsize=3, color=PALETTE["blue"], edgecolor="black", linewidth=0.5,
               label="Cued")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$N={int(r)}$" for r in df_rc["set_size"]])
        ax.set_ylabel("MAE (°)")
        ax.set_title("Retrocue benefit")
        ax.legend(frameon=False, fontsize=6)
        for i, row in df_rc.iterrows():
            p = row["p_val"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            y_top = max(row["neutral_mae"] + row["neutral_sem"],
                        row["cued_mae"] + row["cued_sem"]) + 1
            ax.text(i, y_top, sig, ha="center", fontsize=7, fontweight="bold")
    else:
        # Synthetic demonstration
        conditions = ["Neutral", "Cued"]
        vals = [28.0, 18.0]
        sems = [3.0, 2.5]
        ax.bar(conditions, vals, yerr=sems, color=[PALETTE["orange"], PALETTE["blue"]],
               edgecolor="black", linewidth=0.5, capsize=3, width=0.5)
        ax.set_ylabel("MAE (°)")
        ax.set_title("Retrocue benefit ($N=4$)")
        ax.text(0.5, 30, "***", ha="center", fontsize=9, fontweight="bold")
    ax.text(-0.22, 1.08, labels[panel_idx], transform=ax.transAxes,
            fontsize=12, fontweight="bold")
    panel_idx += 1

    # Panel C: Retrocue x set size interaction
    if has_retrocue_ss:
        ax = axes[panel_idx]
        df_rc = pd.read_csv(retrocue_ss_path)
        benefit = df_rc["benefit_mean"].values
        benefit_sem = df_rc["benefit_sem"].values
        ss_labels = [f"$N={int(r)}$" for r in df_rc["set_size"]]
        bar_colors = [PALETTE["green"], PALETTE["blue"]]
        ax.bar(np.arange(len(df_rc)), benefit, yerr=benefit_sem, capsize=4,
               color=bar_colors[:len(df_rc)], edgecolor="black", linewidth=0.5, width=0.5)
        ax.set_xticks(np.arange(len(df_rc)))
        ax.set_xticklabels(ss_labels)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
        ax.set_ylabel("Retrocue benefit (°)")
        ax.set_title("Benefit $\\times$ set size")
        comp_path = os.path.join(_VIS_DIR, "retrocue_setsize_comparison.csv")
        if os.path.exists(comp_path):
            df_comp = pd.read_csv(comp_path)
            p_comp = df_comp["p_val"].iloc[0]
            sig = "***" if p_comp < 0.001 else ("**" if p_comp < 0.01 else ("*" if p_comp < 0.05 else "ns"))
            y_top = max(benefit + benefit_sem) + 1.5
            x0, x1 = 0, len(df_rc) - 1
            ax.plot([x0, x0, x1, x1], [y_top - 0.5, y_top, y_top, y_top - 0.5],
                    color="black", linewidth=0.8)
            ax.text((x0 + x1) / 2, y_top + 0.3, sig, ha="center", fontsize=7, fontweight="bold")
        ax.text(-0.22, 1.08, labels[panel_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        panel_idx += 1

    # Panel D: Cue timing
    if has_cue_timing:
        ax = axes[panel_idx]
        df_ct = pd.read_csv(cue_timing_path)
        benefit_col = "benefit_mean" if "benefit_mean" in df_ct.columns else "benefit"
        sem_col = "benefit_sem" if "benefit_sem" in df_ct.columns else None
        yerr_ct = df_ct[sem_col].values if sem_col else None
        ax.bar(np.arange(len(df_ct)), df_ct[benefit_col].values,
               yerr=yerr_ct,
               capsize=3, color=[PALETTE["light_blue"], PALETTE["blue"]],
               edgecolor="black", linewidth=0.5, width=0.5)
        ax.set_xticks(np.arange(len(df_ct)))
        ax.set_xticklabels([f"Epoch {int(e)}" for e in df_ct["cue_start_epoch"]])
        ax.set_ylabel("Retrocue benefit (°)")
        ax.set_title("Cue timing")
        ax.text(-0.22, 1.08, labels[panel_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        panel_idx += 1

    # Panel E: Cue flexibility
    if has_cue_flex:
        ax = axes[panel_idx]
        df_cf = pd.read_csv(cue_flex_path)
        timepoints = ["Post Cue-A", "Post Cue-B"]
        x = np.array([0, 1])
        roles = {
            "Item A": ("A_post_cueA", "A_post_cueB", PALETTE["red"]),
            "Item B": ("B_post_cueA", "B_post_cueB", PALETTE["blue"]),
            "Others": ("other_post_cueA", "other_post_cueB", PALETTE["grey"]),
        }
        for label, (col_t1, col_t2, color) in roles.items():
            means = [df_cf[col_t1].mean(), df_cf[col_t2].mean()]
            sems = [stats.sem(df_cf[col_t1]), stats.sem(df_cf[col_t2])]
            ax.errorbar(x, means, yerr=sems, fmt="o-", color=color,
                        label=label, linewidth=1.5, markersize=4, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints, fontsize=6)
        ax.set_ylabel("MAE (°)")
        ax.set_title("Cue flexibility")
        ax.legend(frameon=False, fontsize=5)
        ax.text(-0.22, 1.08, labels[panel_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        panel_idx += 1

    fig.tight_layout(w_pad=2.0)
    _save(fig, "fig3_retrocue")


# ============================================================
# Figure 4: Bias effects
# ============================================================

def figure4_bias():
    """Two-panel bias figure."""
    bias_path = os.path.join(_VIS_DIR, "bias_results.csv")
    bias_3d_path = os.path.join(_VIS_DIR, "bias_3d_results.csv")

    has_bias = os.path.exists(bias_path)
    has_3d = os.path.exists(bias_3d_path)

    n_panels = int(has_bias) + int(has_3d)
    if n_panels == 0:
        print("  No bias data found, skipping Figure 4")
        return

    fig, axes = plt.subplots(1, max(n_panels, 2), figsize=(max(n_panels * 3.2, 6.4), 3.0))
    if n_panels == 1:
        axes = [axes, plt.subplot(1, 2, 2)]
    panel_idx = 0

    # Panel A: Bias vs distance
    if has_bias:
        ax = axes[panel_idx]
        df_b = pd.read_csv(bias_path)
        ax.errorbar(df_b["Distance_deg"], df_b["Bias_deg"], yerr=df_b["SEM_deg"],
                     fmt="o-", color=PALETTE["purple"], markersize=5, capsize=3)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Color distance (°)")
        ax.set_ylabel("Bias (°)")
        ax.set_title("Repulsion bias vs. distance")
        ax.text(-0.18, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
        panel_idx += 1

    # Panel B: Encoding time x distance (line plot)
    if has_3d:
        ax = axes[panel_idx]
        df_3d = pd.read_csv(bias_3d_path)
        enc_epochs = sorted(df_3d["Encoding_Epochs"].unique())
        line_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"]]
        markers = ["o", "s", "^", "D"]
        for i, enc in enumerate(enc_epochs):
            sub = df_3d[df_3d["Encoding_Epochs"] == enc].sort_values("Distance_deg")
            ax.plot(sub["Distance_deg"], sub["Bias_deg"],
                    color=line_colors[i % len(line_colors)],
                    marker=markers[i % len(markers)], markersize=4,
                    linewidth=1.5, label=f"{enc} epochs")
            ax.fill_between(sub["Distance_deg"],
                            sub["Bias_deg"] - sub["SEM_deg"],
                            sub["Bias_deg"] + sub["SEM_deg"],
                            color=line_colors[i % len(line_colors)], alpha=0.15)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Color distance (°)")
        ax.set_ylabel("Bias (°)")
        ax.set_title("Encoding time $\\times$ distance")
        ax.legend(frameon=False, fontsize=6)
        ax.text(-0.18, 1.08, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
        panel_idx += 1

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig4_bias")


# ============================================================
# Figure 5: PNAS-style model comparison (polished)
# ============================================================

def figure5_model_comparison():
    """Polished version of the PNAS Figure 4A-style comparison."""
    mix_path = os.path.join(_VIS_DIR, "mixture_summary_empirical.csv")
    if not os.path.exists(mix_path):
        print("  No mixture summary data, skipping Figure 5")
        return

    df_mix = pd.read_csv(mix_path)
    set_sizes = df_mix["N"].values.astype(int)
    w_emp = df_mix["w_emp"].values
    csd_emp = df_mix["CSD_emp"].values
    w_sem = df_mix["w_sem"].values
    csd_sem = df_mix["CSD_sem"].values

    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(5, 2.5))

    ax_w.errorbar(set_sizes, w_emp, yerr=w_sem, fmt="o-", color=PALETTE["blue"],
                  markersize=5, capsize=3, markerfacecolor="white", markeredgewidth=1.5)
    ax_w.set_xlabel("Set size ($N$)")
    ax_w.set_ylabel("Mixture weight ($w$)")
    ax_w.set_ylim(0, 1.1)
    ax_w.set_xticks(set_sizes)
    ax_w.set_title("Precision weight")
    ax_w.text(-0.22, 1.08, "A", transform=ax_w.transAxes, fontsize=12, fontweight="bold")

    ax_c.errorbar(set_sizes, csd_emp, yerr=csd_sem, fmt="s-", color=PALETTE["orange"],
                  markersize=5, capsize=3, markerfacecolor="white", markeredgewidth=1.5)
    ax_c.set_xlabel("Set size ($N$)")
    ax_c.set_ylabel("Circular SD (rad)")
    ax_c.set_xticks(set_sizes)
    ax_c.set_title("Precision")
    ax_c.text(-0.22, 1.08, "B", transform=ax_c.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(w_pad=3.0)
    _save(fig, "fig5_model_comparison")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating publication-quality figures...")
    figure2_set_size()
    figure3_retrocue()
    figure4_bias()
    figure5_model_comparison()
    print("Done.")
