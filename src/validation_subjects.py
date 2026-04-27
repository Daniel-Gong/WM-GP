"""
validation_subjects.py
======================
Subject-level validation harness for MemGP experiments.

Runs any of the three experiments – set-size, retrocue, bias – across
multiple simulated "subjects" (each subject uses a different random seed
base, so their individual trials are statistically independent).

After collecting per-subject summary stats the script produces the same
plots as validation.py but with **error bars across subject means** (SEM
over subjects) instead of within-subject SEM/SD.

Usage examples
--------------
# Set-size experiment, 10 model runs, 100 trials each
python validation_subjects.py set_size \
    --config config/config_set_size.yaml \
    --n_subjects 10 \
    --n_trials 100 \
    --save_dir visualizations/subjects/set_size

# Retrocue experiment
python validation_subjects.py retrocue \
    --config config/config_retrocue.yaml \
    --n_subjects 10 \
    --n_trials 100 \
    --save_dir visualizations/subjects/retrocue

# Bias experiment
python validation_subjects.py bias \
    --config config/config_bias.yaml \
    --n_subjects 10 \
    --n_trials 100 \
    --save_dir visualizations/subjects/bias
"""

import argparse
import os
import copy

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib as mpl

import yaml
import generator
from simulation import run_simulation_trial, retrieve_color
import viz.visualizations as vis

# Repo root (parent of `src/`); relative save_dir values are resolved here, not against cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs_save_dir(save_dir, *default_segments):
    """
    If save_dir is None, use default_segments joined as a path under the repo root.
    If save_dir is relative, interpret it under the repo root (not cwd).
    Absolute save_dir values are normalized and left unchanged.
    """
    rel = os.path.join(*default_segments) if save_dir is None else save_dir
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(_REPO_ROOT, rel))


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path=None, filename="config.yaml"):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _patch_config(base_config: dict, n_trials: int, subject_idx: int) -> dict:
    """Return a deep-copied config with n_trials and a subject-unique random seed."""
    cfg = copy.deepcopy(base_config)
    cfg["experiment"]["n_trials"] = n_trials
    # shift the base seed by subject index * large prime so seeds don't overlap
    cfg["experiment"]["random_seed"] = base_config["experiment"]["random_seed"] + subject_idx * 100003
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject runners  (mirrors validation.py but returns raw arrays)
# ──────────────────────────────────────────────────────────────────────────────

def _subject_set_size(config: dict):
    """
    Run one subject's set-size sweep.

    Returns
    -------
    dict  {set_size: {'mean_abs_err': float, 'sd_signed': float}}
    """
    result = {}
    for n_items in config["experiment"]["set_sizes"]:
        all_signed = []
        for t in trange(config["experiment"]["n_trials"],
                        desc=f"  Set Size {n_items}", leave=False):
            seed = config["experiment"]["random_seed"] + t
            items = generator.generate_items(n_items, seed=seed)
            model, _, _ = run_simulation_trial(
                items, config, cued_item_idx=None, track_visuals=False
            )
            for i in range(len(items)):
                s_err = retrieve_color(model, items[i][0], items[i][1])
                all_signed.append(s_err)

        result[n_items] = {
            "mean_abs_err": float(np.mean(np.abs(all_signed))),
            "sd_signed":    float(np.std(all_signed)),
        }
    return result


def _subject_retrocue(config: dict, target_set_size: int = 4):
    """
    Run one subject's retrocue experiment.

    Returns
    -------
    dict  {'mean_neutral': float, 'mean_cued': float}
    """
    cued_errors    = []
    neutral_errors = []

    for t in trange(config["experiment"]["n_trials"],
                    desc="  Retrocue", leave=False):
        seed = config["experiment"]["random_seed"] + t
        items = generator.generate_items(target_set_size, seed=seed)
        probed_idx = np.random.randint(0, target_set_size)

        # Neutral (no cue)
        model, _, _ = run_simulation_trial(items, config, cued_item_idx=None)
        neutral_errors.append(
            abs(retrieve_color(model, items[probed_idx][0], items[probed_idx][1]))
        )

        # Cued
        model, _, _ = run_simulation_trial(items, config, cued_item_idx=probed_idx)
        cued_errors.append(
            abs(retrieve_color(model, items[probed_idx][0], items[probed_idx][1]))
        )

    return {
        "mean_neutral": float(np.mean(neutral_errors)),
        "mean_cued":    float(np.mean(cued_errors)),
    }


def _subject_bias(
    config: dict,
    distances=(20, 45, 90, 135),
    encoding_epochs_list=(50, 100, 150, 200),
    nt_loc=90.0,
):
    """
    Run one subject's bias experiment across distance × encoding epochs.

    Returns
    -------
    list of dicts, each with keys:
        Distance_deg, Encoding_Epochs, Bias_pct, Bias_deg
    """
    t_loc = 0.0
    normalization = config.get("normalization", None)
    n_trials = config["experiment"]["n_trials"]
    results = []

    for enc_epochs in encoding_epochs_list:
        for dist in distances:
            signed_errors = []
            desc = f"  Enc={enc_epochs}, Dist={dist}°"

            for _ in trange(n_trials, desc=desc, leave=False):
                t_col = int(np.random.uniform(-180.0, 180.0))
                nt_col = (t_col + dist + 180.0) % 360.0 - 180.0
                items = [(t_loc, t_col), (nt_loc, nt_col)]

                model, _, _ = run_simulation_trial(
                    items, config, encoding_epochs=enc_epochs,
                )
                s_err = retrieve_color(
                    model, t_loc, t_col, normalization=normalization,
                )
                signed_errors.append(s_err)

            num_repulsion = sum(1 for e in signed_errors if e < 0)
            bias_pct = (num_repulsion / len(signed_errors)) * 100 - 50

            results.append({
                "Distance_deg": dist,
                "Encoding_Epochs": enc_epochs,
                "Bias_pct": bias_pct,
                "Bias_deg": float(np.mean(signed_errors)),
            })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Across-subject aggregation & plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_set_size_effect_subjects(subject_records, save_dir=None):
    """
    Parameters
    ----------
    subject_records : list of dicts returned by _subject_set_size()
        Each dict maps set_size -> {'mean_abs_err': float, ...}
    """
    save_dir = _abs_save_dir(save_dir, "visualizations")
    set_sizes = sorted(subject_records[0].keys())
    n_subj    = len(subject_records)

    means_arr = np.array([[r[ss]["mean_abs_err"] for ss in set_sizes]
                          for r in subject_records])   # (n_subj, n_set_sizes)

    group_mean = means_arr.mean(axis=0)
    group_sem  = stats.sem(means_arr, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Individual subject lines (light grey, thin)
    for subj_row in means_arr:
        ax.plot(set_sizes, subj_row, color="grey", linewidth=0.8, alpha=0.4)

    # Group mean ± SEM
    ax.errorbar(
        set_sizes, group_mean, yerr=group_sem,
        marker="o", linestyle="-", color="steelblue", markersize=9,
        capsize=6, capthick=1.8, elinewidth=1.8, ecolor="steelblue",
        label=f"Group mean ± SEM  (N={n_subj} subjects)",
    )

    ax.set_title("Set Size Effect on Retrieval Error\n(across subjects)", fontsize=13)
    ax.set_xlabel("Set Size (N)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (°)", fontsize=12)
    ax.set_xticks(set_sizes)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "set_size_effect_subjects.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_retrocue_benefit_subjects(subject_records, target_set_size=4, save_dir=None):
    """
    Parameters
    ----------
    subject_records : list of dicts with keys 'mean_neutral', 'mean_cued'
    """
    save_dir = _abs_save_dir(save_dir, "visualizations")
    neutral_arr = np.array([r["mean_neutral"] for r in subject_records])
    cued_arr    = np.array([r["mean_cued"]    for r in subject_records])
    n_subj      = len(subject_records)

    group_mean = [neutral_arr.mean(), cued_arr.mean()]
    group_sem  = [stats.sem(neutral_arr), stats.sem(cued_arr)]

    t_stat, p_val = stats.ttest_rel(neutral_arr, cued_arr)

    conditions = ["Neutral", "Cued"]
    colors     = ["lightcoral", "lightgreen"]

    fig, ax = plt.subplots(figsize=(5, 5))

    bars = ax.bar(conditions, group_mean, yerr=group_sem,
                  color=colors, edgecolor="black", width=0.5, capsize=6)

    # Overlay individual subject dots + lines
    for n_val, c_val in zip(neutral_arr, cued_arr):
        ax.plot([0, 1], [n_val, c_val], color="grey", linewidth=0.8, alpha=0.4)
        ax.plot([0, 1], [n_val, c_val], marker="o", color="grey",
                markersize=4, linewidth=0, alpha=0.5)

    # Value annotations
    for i, (v, se) in enumerate(zip(group_mean, group_sem)):
        ax.text(i, v + se + max(0.3, v * 0.04), f"{v:.2f}",
                ha="center", fontweight="bold", fontsize=10)

    # Significance bracket
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "ns"

    y_max = max(group_mean[i] + group_sem[i] for i in range(2))
    h     = max(0.8, 0.05 * y_max)
    ax.plot([0, 0, 1, 1], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h],
            lw=1.5, c="k")
    ax.text(0.5, y_max + 2.2*h, sig, ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    ax.set_ylim(0, y_max + 5*h)

    ax.set_title(
        f"Retrocue Benefit (N={target_set_size} items)\n"
        f"Paired t={t_stat:.2f}, p={p_val:.3e} | {n_subj} subjects",
        fontsize=11,
    )
    ax.set_ylabel("Mean Absolute Error (°)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"retrocue_benefit_subjects_N{target_set_size}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_bias_effect_subjects(df_all, save_dir=None):
    """
    Multi-line plots of bias vs color distance (one line per encoding epoch),
    with ±1 SEM bands computed across subjects.

    Produces two figures: one for Bias_pct and one for Bias_deg.

    Parameters
    ----------
    df_all : pd.DataFrame
        Must contain columns: Subject, Distance_deg, Encoding_Epochs,
        Bias_pct, Bias_deg.
    """
    save_dir = _abs_save_dir(save_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    distances = sorted(df_all["Distance_deg"].unique())
    enc_epochs = sorted(df_all["Encoding_Epochs"].unique())
    n_subj = df_all["Subject"].nunique()

    line_colors = ["#4A90D9", "#E8834A", "#4CAF50", "#800080"]
    markers = ["o", "s", "^", "D"]

    for metric, ylabel, suffix in [
        ("Bias_pct", "% Repulsion Bias", "pct"),
        ("Bias_deg", "Bias (°)", "deg"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))

        for i, enc in enumerate(enc_epochs):
            grp = df_all[df_all["Encoding_Epochs"] == enc]
            agg = grp.groupby("Distance_deg")[metric]
            means = agg.mean()
            sems = agg.apply(stats.sem)

            c = line_colors[i % len(line_colors)]
            ax.plot(
                means.index, means.values,
                color=c, marker=markers[i % len(markers)],
                markersize=8, linewidth=2,
                label=f"{enc} epochs",
            )
            ax.fill_between(
                means.index,
                means.values - sems.values,
                means.values + sems.values,
                color=c, alpha=0.18,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(distances)
        ax.set_xlabel("Color Distance (°)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(
            f"Bias vs Color Distance  (N={n_subj} subjects, ±1 SEM)",
            fontsize=12,
        )
        ax.legend(fontsize=11, framealpha=0.9)
        ax.tick_params(labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        out = os.path.join(save_dir, f"bias_line_subjects_{suffix}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


def plot_collapsed_bias_bars_subjects(df_all, epoch_levels=None, save_dir=None):
    """
    Bar plots of bias collapsed across color distances, with SEM across subjects.

    Produces two figures: % repulsion bias and repulsion in degrees (negated
    so positive = repulsion).

    Parameters
    ----------
    df_all : pd.DataFrame
        Must contain columns: Subject, Distance_deg, Encoding_Epochs,
        Bias_pct, Bias_deg.
    """
    save_dir = _abs_save_dir(save_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    if epoch_levels is None:
        epoch_levels = tuple(sorted(df_all["Encoding_Epochs"].unique()))

    sub = df_all[df_all["Encoding_Epochs"].isin(epoch_levels)].copy()
    n_subj = sub["Subject"].nunique()

    collapsed = (
        sub.groupby(["Subject", "Encoding_Epochs"])
        .agg(Bias_pct=("Bias_pct", "mean"), Bias_deg=("Bias_deg", "mean"))
        .reset_index()
    )

    bar_colors = ["#4A90D9", "#E8834A", "#4CAF50", "#800080"]
    x = np.arange(len(epoch_levels))
    width = 0.55

    for metric, ylabel, unit, suffix, sign in [
        ("Bias_pct", "% Repulsion Bias", "%", "pct", 1),
        ("Bias_deg", "Repulsion Bias (°)", "°", "deg", -1),
    ]:
        means, sems = [], []
        for enc in epoch_levels:
            vals = collapsed.loc[
                collapsed["Encoding_Epochs"] == enc, metric
            ].values * sign
            means.append(vals.mean())
            sems.append(stats.sem(vals))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(
            x, means, width, yerr=sems, capsize=6,
            color=bar_colors[: len(epoch_levels)], edgecolor="black",
            linewidth=0.8, error_kw=dict(lw=1.5),
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in epoch_levels], fontsize=12)
        ax.set_xlabel("Encoding Epochs", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(
            f"Repulsion Bias ({unit}) by Encoding Epochs\n"
            f"(N={n_subj} subjects, collapsed across distances)",
            fontsize=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)

        for i, (m, s) in enumerate(zip(means, sems)):
            offset = s + (1.5 if suffix == "pct" else 0.3)
            ax.text(
                i, m + offset, f"{m:.1f}{unit}",
                ha="center", fontsize=10, fontweight="bold",
            )

        fig.tight_layout()
        out = os.path.join(save_dir, f"bias_collapsed_subjects_{suffix}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# High-level experiment runners (subject loop)
# ──────────────────────────────────────────────────────────────────────────────

def run_set_size_experiment_subjects(
    base_config: dict,
    n_subjects: int = 10,
    n_trials: int = 100,
    save_dir=None,
):
    save_dir = _abs_save_dir(save_dir, "visualizations", "subjects", "set_size")
    print(f"\n=== Set Size Experiment — {n_subjects} subjects × {n_trials} trials ===")
    subject_records = []

    for subj in range(n_subjects):
        print(f"\n── Subject {subj + 1}/{n_subjects} ──")
        cfg    = _patch_config(base_config, n_trials, subj)
        record = _subject_set_size(cfg)
        subject_records.append(record)

    # Save per-subject CSV
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for subj_idx, rec in enumerate(subject_records):
        for ss, vals in rec.items():
            rows.append({"Subject": subj_idx, "Set Size": ss, **vals})
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(save_dir, "set_size_subjects_raw.csv"), index=False)

    # Group summary
    group_rows = []
    for ss in sorted(subject_records[0].keys()):
        means = [r[ss]["mean_abs_err"] for r in subject_records]
        group_rows.append({
            "Set Size": ss,
            "Group Mean Abs Error": np.mean(means),
            "Group SEM Abs Error":  stats.sem(means),
        })
    df_group = pd.DataFrame(group_rows)
    df_group.to_csv(os.path.join(save_dir, "set_size_group_summary.csv"), index=False)
    print("\nGroup summary:")
    print(df_group.to_string(index=False))

    plot_set_size_effect_subjects(subject_records, save_dir=save_dir)
    return subject_records


def run_retrocue_experiment_subjects(
    base_config: dict,
    target_set_size: int = 4,
    n_subjects: int = 10,
    n_trials: int = 100,
    save_dir=None,
):
    save_dir = _abs_save_dir(save_dir, "visualizations", "subjects", "retrocue")
    print(f"\n=== Retrocue Experiment — {n_subjects} subjects × {n_trials} trials (N={target_set_size}) ===")
    subject_records = []

    for subj in range(n_subjects):
        print(f"\n── Subject {subj + 1}/{n_subjects} ──")
        cfg    = _patch_config(base_config, n_trials, subj)
        record = _subject_retrocue(cfg, target_set_size=target_set_size)
        subject_records.append(record)
        print(f"   Neutral: {record['mean_neutral']:.3f}°  |  Cued: {record['mean_cued']:.3f}°  |  Benefit: {record['mean_neutral'] - record['mean_cued']:.3f}°")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(subject_records)
    df.insert(0, "Subject", range(len(subject_records)))
    df["Benefit"] = df["mean_neutral"] - df["mean_cued"]
    df.to_csv(os.path.join(save_dir, "retrocue_subjects.csv"), index=False)
    print("\nGroup summary:")
    print(df[["mean_neutral", "mean_cued", "Benefit"]].describe().to_string())

    plot_retrocue_benefit_subjects(
        subject_records,
        target_set_size=target_set_size,
        save_dir=save_dir,
    )
    return subject_records


def run_bias_experiment_subjects(
    base_config: dict,
    n_subjects: int = 10,
    n_trials: int = 100,
    distances=(20, 45, 90, 135),
    encoding_epochs_list=(50, 100, 150, 200),
    nt_loc: float = 90.0,
    save_dir=None,
):
    save_dir = _abs_save_dir(save_dir, "visualizations", "subjects", "bias")
    n_cond = len(distances) * len(encoding_epochs_list)
    print(
        f"\n=== Bias Experiment — {n_subjects} subjects × {n_trials} trials "
        f"× {n_cond} conditions ==="
    )

    all_rows = []
    for subj in range(n_subjects):
        print(f"\n── Subject {subj + 1}/{n_subjects} ──")
        cfg = _patch_config(base_config, n_trials, subj)
        records = _subject_bias(
            cfg,
            distances=distances,
            encoding_epochs_list=encoding_epochs_list,
            nt_loc=nt_loc,
        )
        for rec in records:
            rec["Subject"] = subj
            all_rows.append(rec)

    df_all = pd.DataFrame(all_rows)

    os.makedirs(save_dir, exist_ok=True)
    df_all.to_csv(os.path.join(save_dir, "bias_subjects_raw.csv"), index=False)

    # Group summary: mean ± SEM across subjects per condition
    group = (
        df_all.groupby(["Encoding_Epochs", "Distance_deg"])
        .agg(
            Mean_Bias_pct=("Bias_pct", "mean"),
            SEM_Bias_pct=("Bias_pct", lambda x: stats.sem(x)),
            Mean_Bias_deg=("Bias_deg", "mean"),
            SEM_Bias_deg=("Bias_deg", lambda x: stats.sem(x)),
        )
        .reset_index()
    )
    group.to_csv(os.path.join(save_dir, "bias_group_summary.csv"), index=False)
    print("\nGroup summary:")
    print(group.to_string(index=False))

    plot_bias_effect_subjects(df_all, save_dir=save_dir)
    plot_collapsed_bias_bars_subjects(
        df_all, epoch_levels=encoding_epochs_list, save_dir=save_dir,
    )
    return df_all


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MemGP experiments across multiple simulated subjects."
    )
    parser.add_argument(
        "experiment",
        choices=["set_size", "retrocue", "bias"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML config file. "
            "Defaults: set_size→config/config_set_size.yaml, "
            "retrocue→config/config_retrocue.yaml, bias→config/config_bias.yaml"
        ),
    )
    parser.add_argument("--n_subjects", type=int, default=50,
                        help="Number of simulated subjects (default: 50).")
    parser.add_argument("--n_trials",   type=int, default=100,
                        help="Trials per condition per subject (default: 100).")
    parser.add_argument("--set_size",   type=int, default=2,
                        help="Set size for retrocue experiment (default: 2).")
    parser.add_argument("--save_dir",   default=None,
                        help="Output directory (default: visualizations/subjects/<experiment>).")
    args = parser.parse_args()

    # Default config filenames
    default_configs = {
        "set_size": "config_set_size.yaml",
        "retrocue": "config_retrocue.yaml",
        "bias":     "config_bias.yaml",
    }
    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "config", default_configs[args.experiment]
    )
    base_config = load_config(path=config_path)

    save_dir = _abs_save_dir(
        args.save_dir, "visualizations", "subjects", args.experiment
    )

    if args.experiment == "set_size":
        run_set_size_experiment_subjects(
            base_config,
            n_subjects=args.n_subjects,
            n_trials=args.n_trials,
            save_dir=save_dir,
        )
    elif args.experiment == "retrocue":
        run_retrocue_experiment_subjects(
            base_config,
            target_set_size=args.set_size,
            n_subjects=args.n_subjects,
            n_trials=args.n_trials,
            save_dir=save_dir,
        )
    elif args.experiment == "bias":
        run_bias_experiment_subjects(
            base_config,
            n_subjects=args.n_subjects,
            n_trials=args.n_trials,
            distances=(20, 45, 90, 135),
            encoding_epochs_list=(50, 100, 150, 200),
            nt_loc=90.0,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
