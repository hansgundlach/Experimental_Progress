"""
Analyze LR sweep results: find best LR per dimension, fit scaling law, plot, and
output a table of fitted optimal LRs for hidden_dims 32-512 (step 16).

Usage:
    python analyze_lr.py --group modern_transformer
    python analyze_lr.py --group modern_transformer --all-groups
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir for imports if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lr_experiment_groups import LR_EXPERIMENT_GROUPS, expand_subgroups, is_combined_group

# Param-count helpers (Kaplan et al. 2020 non-embedding-N convention).
from experiment_utils import calculate_non_embedding_params
try:
    from LSTM_model.lstm_experiment_utils import calculate_lstm_params
except Exception:
    calculate_lstm_params = None


def _resolve_concrete_group(group_name):
    """Return the concrete group config used to derive arch/overrides.

    For a `combine`d group, returns the first subgroup's config (all subgroups
    in our defs share the same architecture/overrides, only LR/dim grids
    differ). Returns None if the name is not in LR_EXPERIMENT_GROUPS.
    """
    g = LR_EXPERIMENT_GROUPS.get(group_name)
    if g is None:
        return None
    if "combine" in g:
        for sub in g["combine"]:
            sg = LR_EXPERIMENT_GROUPS.get(sub)
            if sg and "combine" not in sg:
                return sg
        return None
    return g


def params_for_dim(hidden_dim, group_config):
    """Compute non-embedding parameter count N for a given hidden_dim.

    Mirrors `experiment_utils.gen_experim` for the transformer case
    (num_layers = max(1, round(hidden_dim / 16)), num_heads chosen by
    target head dim) and uses `calculate_lstm_params` for LSTMs. Returns
    None if architecture-specific helpers can't be applied.
    """
    if group_config is None:
        return None
    arch = group_config.get("architecture", "transformer")
    overrides = group_config.get("model_overrides", {}) or {}
    if arch == "lstm":
        if calculate_lstm_params is None:
            return None
        num_layers = overrides.get("num_layers", 2)
        tie = overrides.get("tie_embeddings", True)
        result = calculate_lstm_params(
            hidden_size=hidden_dim, num_layers=num_layers, tie_embeddings=tie,
        )
        # Subtract embedding portion to match Kaplan-style non-embedding N.
        if isinstance(result, dict):
            total = result.get("trainable_params", result.get("total_params"))
        else:
            total = int(result)
        vocab = 50257
        return int(total - vocab * hidden_dim) if total is not None else None
    # Transformer: replicate gen_experim's depth/heads rule.
    base_hidden_dim, base_num_layers = 32, 2
    layer_scale_ratio = base_num_layers / base_hidden_dim
    num_layers = max(1, int(round(hidden_dim * layer_scale_ratio)))
    return int(calculate_non_embedding_params(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=overrides.get("activation", "swiglu"),
        vocab_size=50257,
        seq_length=overrides.get("seq_length", 128),
        pos_encoding=overrides.get("pos_encoding", "rotary"),
        tie_embeddings=overrides.get("tie_embeddings", True),
        ff_ratio=overrides.get("ff_ratio", 4),
    ))


def fit_lr_scaling_vs_N(parabolic_fit, group_config):
    """Fit log10(eta*) = alpha * log10(N) + beta using the per-dim estimator
    chosen in fit_lr_scaling_parabolic (vertex when valid, argmin fallback
    otherwise). N is non-embedding parameter count (Kaplan et al. 2020).

    Within a single architecture this is the literature-standard axis
    (Chinchilla, Kaplan, Bjorck 2025, Hägele 2024). Returns None if N can't
    be computed (e.g. unknown group config) or fewer than 2 dims yield N.
    """
    if parabolic_fit is None:
        return None
    per_dim_est = parabolic_fit.get("per_dim_estimator", {})
    if not per_dim_est:
        return None
    pairs = []
    for d, (_estimator, eta_used, _reason) in per_dim_est.items():
        N = params_for_dim(int(d), group_config)
        if N is not None and N > 0 and np.isfinite(eta_used) and eta_used > 0:
            pairs.append((int(d), int(N), float(eta_used)))
    if len(pairs) < 2:
        return None
    pairs.sort()
    Ns = np.array([p[1] for p in pairs], dtype=float)
    lrs = np.array([p[2] for p in pairs], dtype=float)
    log_N = np.log10(Ns)
    log_lr = np.log10(lrs)
    slope, intercept = np.polyfit(log_N, log_lr, 1)
    pred = slope * log_N + intercept
    ss_res = float(np.sum((log_lr - pred) ** 2))
    ss_tot = float(np.sum((log_lr - np.mean(log_lr)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "alpha_N": float(slope),
        "beta_N": float(intercept),
        "r2_N": float(r2),
        "dims": [p[0] for p in pairs],
        "Ns": [p[1] for p in pairs],
        "etas_used": [p[2] for p in pairs],
    }


def make_parabolic_scaling_vs_N_plot(group_name, parabolic_fit, fit_N, group_config, plots_dir):
    """log10(eta*) vs log10(N), color-coded by per-dim estimator. Mirrors the
    d-axis plot but on the literature-standard parameter-count axis (Kaplan
    2020 / Bjorck 2025)."""
    if fit_N is None or parabolic_fit is None:
        return
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    per_dim_est = parabolic_fit.get("per_dim_estimator", {})
    plotted_vertex = plotted_argmin = False
    for d in sorted(per_dim_est.keys()):
        est_name, eta_used, _ = per_dim_est[d]
        N = params_for_dim(int(d), group_config)
        if N is None or N <= 0 or not np.isfinite(eta_used) or eta_used <= 0:
            continue
        if est_name == "vertex":
            ax.scatter(N, eta_used, c="tab:blue", s=120,
                       edgecolors="black", linewidth=1.2, zorder=3,
                       label="η* (parabolic vertex)" if not plotted_vertex else None)
            plotted_vertex = True
        else:
            ax.scatter(N, eta_used, c="tab:orange", marker="s", s=120,
                       edgecolors="black", linewidth=1.2, zorder=3,
                       label="η* (argmin fallback)" if not plotted_argmin else None)
            plotted_argmin = True
        ax.annotate(f"d={d}", (N, eta_used), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, color="dimgray")

    a = fit_N["alpha_N"]; b = fit_N["beta_N"]; r2 = fit_N["r2_N"]
    Ns_arr = np.array(fit_N["Ns"], dtype=float)
    x_fit = np.logspace(np.log10(Ns_arr.min()), np.log10(Ns_arr.max()), 200)
    y_fit = 10 ** (a * np.log10(x_fit) + b)
    ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.85,
            label=f"η* = {10**b:.3e} · N^({a:.3f})  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Non-embedding parameters N", fontsize=14)
    ax.set_ylabel("η*  (parabolic vertex / argmin fallback)", fontsize=14)
    ax.set_title(f"LR scaling vs N: {group_name}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    png_path = os.path.join(plots_dir, f"{group_name}_parabolic_scaling_vs_N.png")
    pdf_path = os.path.join(plots_dir, f"{group_name}_parabolic_scaling_vs_N.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


def make_d_vs_N_comparison_plot(group_name, parabolic_fit, fit_N, group_config, plots_dir):
    """Side-by-side: log10(eta*) vs log10(d) and vs log10(N) with both fits.
    Lets you eyeball which axis is straighter."""
    if fit_N is None or parabolic_fit is None:
        return
    os.makedirs(plots_dir, exist_ok=True)
    per_dim_est = parabolic_fit.get("per_dim_estimator", {})
    fig, (ax_d, ax_N) = plt.subplots(1, 2, figsize=(14, 5))

    rows = []
    for d in sorted(per_dim_est.keys()):
        est_name, eta_used, _ = per_dim_est[d]
        N = params_for_dim(int(d), group_config)
        if N is None or N <= 0 or not np.isfinite(eta_used) or eta_used <= 0:
            continue
        rows.append((int(d), int(N), float(eta_used), est_name))
    if not rows:
        plt.close(fig); return
    ds = np.array([r[0] for r in rows], dtype=float)
    Ns = np.array([r[1] for r in rows], dtype=float)
    etas = np.array([r[2] for r in rows], dtype=float)
    colors = ["tab:blue" if r[3] == "vertex" else "tab:orange" for r in rows]
    markers = ["o" if r[3] == "vertex" else "s" for r in rows]

    for ax, xs, label in [(ax_d, ds, "Hidden dimension d"),
                          (ax_N, Ns, "Non-embedding parameters N")]:
        for x, y, c, m in zip(xs, etas, colors, markers):
            ax.scatter(x, y, c=c, marker=m, s=120, edgecolors="black",
                       linewidth=1.2, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel("η*", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")

    a_d = parabolic_fit["alpha"]; b_d = parabolic_fit["beta"]; r2_d = parabolic_fit["r2"]
    x_d = np.logspace(np.log10(ds.min()), np.log10(ds.max()), 200)
    ax_d.plot(x_d, 10 ** (a_d * np.log10(x_d) + b_d), "r--", linewidth=2,
              label=f"η* = {10**b_d:.3e}·d^({a_d:.3f})  R²={r2_d:.3f}")
    ax_d.set_title(f"vs d (slope {a_d:.3f}, R²={r2_d:.3f})", fontsize=12)
    ax_d.legend(fontsize=10)

    a_N = fit_N["alpha_N"]; b_N = fit_N["beta_N"]; r2_N = fit_N["r2_N"]
    x_N = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 200)
    ax_N.plot(x_N, 10 ** (a_N * np.log10(x_N) + b_N), "r--", linewidth=2,
              label=f"η* = {10**b_N:.3e}·N^({a_N:.3f})  R²={r2_N:.3f}")
    ax_N.set_title(f"vs N (slope {a_N:.3f}, R²={r2_N:.3f})", fontsize=12)
    ax_N.legend(fontsize=10)

    fig.suptitle(f"{group_name}: d-axis vs N-axis fit", fontsize=14, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(plots_dir, f"{group_name}_d_vs_N_comparison.png")
    pdf_path = os.path.join(plots_dir, f"{group_name}_d_vs_N_comparison.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


def write_parabolic_vs_N_csv(group_name, fit_N, group_config, target_dims, plots_dir):
    """Emit hidden_dim-indexed extrapolation table using the N-axis fit."""
    if fit_N is None:
        print("  [N-axis] Could not compute N for this group; skipping.")
        return
    os.makedirs(plots_dir, exist_ok=True)
    a, b, r2 = fit_N["alpha_N"], fit_N["beta_N"], fit_N["r2_N"]
    table_path = os.path.join(plots_dir, "optimal_lrs_parabolic_vs_N.csv")
    with open(table_path, "w") as f:
        f.write(
            f"# {group_name}: parabolic+argmin scaling law vs non-embedding N\n"
            f"# Fit: log10(eta*) = {a:.4f} * log10(N) + {b:.4f},  R^2 = {r2:.4f}\n"
            f"# Equivalently: eta* = {10**b:.6e} * N^({a:.4f})\n"
            f"# N is computed per hidden_dim via experiment_utils.calculate_non_embedding_params\n"
            f"# (Kaplan et al. 2020 definition; matches Bjorck 2025 / Chinchilla protocol).\n"
            "hidden_dim,N_non_embedding,learning_rate,log10_lr\n"
        )
        for d in target_dims:
            N = params_for_dim(int(d), group_config)
            if N is None or N <= 0:
                continue
            lr = 10 ** (a * np.log10(N) + b)
            f.write(f"{d},{N},{lr:.8f},{np.log10(lr):.4f}\n")
    print(f"  Table saved: {table_path}")


def print_parabolic_vs_N_results(group_name, fit_N, group_config, target_dims):
    if fit_N is None:
        print(f"\n  ---- N-AXIS FIT : {group_name} ----")
        print("  Skipped: could not compute N for this group config.")
        return
    a, b, r2 = fit_N["alpha_N"], fit_N["beta_N"], fit_N["r2_N"]
    print(f"\n  ---- N-AXIS FIT : {group_name} ----")
    print(f"  Fit (using vertex+argmin per-dim estimators):")
    print(f"    log10(eta*) = {a:.4f} * log10(N) + {b:.4f}    R²={r2:.4f}")
    print(f"    eta* = {10**b:.6e} * N^({a:.4f})")
    print(f"    N is non-embedding params (Kaplan 2020 def); hidden_dim->N via gen_experim rule.")
    print(f"\n  Per-dim used in fit:")
    print(f"    {'dim':<6} {'N':<14} {'eta_used':<14}")
    for d, N, eta in zip(fit_N["dims"], fit_N["Ns"], fit_N["etas_used"]):
        print(f"    {d:<6} {N:<14,d} {eta:<.4e}")
    print(f"\n  Predicted η* by hidden_dim (drop-in for experiment_definitions.py):")
    print(f"    {'dim':<6} {'N':<14} {'eta_pred':<14} {'log10':<8}")
    pred_dict = {}
    for d in target_dims:
        N = params_for_dim(int(d), group_config)
        if N is None or N <= 0:
            continue
        lr = 10 ** (a * np.log10(N) + b)
        pred_dict[int(d)] = lr
        print(f"    {d:<6} {N:<14,d} {lr:<.4e}      {np.log10(lr):.2f}")
    if pred_dict:
        items = ", ".join(f"{k}: {v:.6f}" for k, v in pred_dict.items())
        print(f"\n  # {group_name}: vs-N scaling, eta* = {10**b:.4e} * N^({a:.4f}), R²={r2:.3f}")
        print(f"  OPTIMAL_LRS_VS_N = {{{items}}}")


def parse_lr_csv_name(filename):
    """Extract (hidden_dim, learning_rate) from a CSV filename like '64d_lr_3e-02.csv'."""
    base = filename.replace(".csv", "")
    dim_match = re.search(r"(\d+)d", base)
    if not dim_match:
        return None, None
    dim = int(dim_match.group(1))

    lr_match = re.search(r"lr_(.+)$", base)
    if not lr_match:
        return dim, None
    lr_str = lr_match.group(1)

    # Current convention: a bare number (no 'e') is the log10(lr) exponent.
    #   e.g. "-2"    -> 10^-2    = 0.01
    #        "-1.25" -> 10^-1.25 = 0.0562
    if "e" not in lr_str:
        try:
            return dim, 10.0 ** float(lr_str)
        except ValueError:
            return dim, None

    # Legacy: old writer emitted "10e{N}" meaning 10^N (e.g. "10e-2" for 0.01),
    # but float("10e-2") is standard sci notation = 0.1. Correct it.
    legacy_match = re.fullmatch(r"10e([+-]?\d+)", lr_str)
    if legacy_match:
        return dim, 10.0 ** int(legacy_match.group(1))

    # Otherwise treat as standard scientific notation (e.g. "3e-2", "1.8e-2").
    try:
        return dim, float(lr_str)
    except ValueError:
        return dim, None


def get_final_val_loss(csv_path):
    """Read a training CSV and return the final validation loss."""
    try:
        df = pd.read_csv(csv_path)
        val_cols = [c for c in df.columns if "validation" in c.lower() and "loss" in c.lower()]
        if not val_cols:
            return None
        losses = df[val_cols[0]].dropna()
        if len(losses) == 0:
            return None
        return float(losses.iloc[-1])
    except Exception:
        return None


def collect_records(results_dirs):
    """
    Scan one or more results dirs and return a flat list of (dim, lr, loss) records.
    """
    if isinstance(results_dirs, (str, os.PathLike)):
        results_dirs = [results_dirs]

    records = []
    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            print(f"Results directory not found: {results_dir}")
            continue
        csv_files = sorted(Path(results_dir).glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {results_dir}")
            continue
        for csv_path in csv_files:
            dim, lr = parse_lr_csv_name(csv_path.name)
            if dim is None or lr is None:
                continue
            loss = get_final_val_loss(str(csv_path))
            if loss is None or np.isnan(loss):
                continue
            records.append((dim, lr, loss))
    return records


def find_best_lr_per_dim(results_dirs):
    """
    Scan one or more results dirs and return {dim: (best_lr, best_loss, all_results)}.
    all_results is a list of (lr, loss) for that dim.

    `results_dirs` may be a single path (concrete group) or a list of paths
    (combined group — records from all dirs are pooled by dim).
    """
    records = collect_records(results_dirs)
    if not records:
        return {}

    # Group by dimension (records from different subgroups merge by dim if they overlap).
    from collections import defaultdict
    by_dim = defaultdict(list)
    for dim, lr, loss in records:
        by_dim[dim].append((lr, loss))

    best = {}
    for dim, results in sorted(by_dim.items()):
        results.sort(key=lambda x: x[1])
        best_lr, best_loss = results[0]
        best[dim] = (best_lr, best_loss, results)

    return best


def fit_lr_scaling(best_per_dim):
    """
    Fit log10(lr) = a + b * log10(dim) to the best LRs.
    Returns (a, b, r_squared) or None if fit fails.
    """
    dims = np.array(sorted(best_per_dim.keys()), dtype=float)
    lrs = np.array([best_per_dim[int(d)][0] for d in dims])

    if len(dims) < 2:
        print("Need at least 2 dimensions to fit scaling law")
        return None

    log_dims = np.log10(dims)
    log_lrs = np.log10(lrs)

    # Linear fit in log-log space
    coeffs = np.polyfit(log_dims, log_lrs, 1)
    b, a = coeffs  # slope, intercept

    # R-squared
    predicted = a + b * log_dims
    ss_res = np.sum((log_lrs - predicted) ** 2)
    ss_tot = np.sum((log_lrs - np.mean(log_lrs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return a, b, r_squared


def extrapolate_lrs(a, b, dims):
    """Given fit params, return {dim: lr} for each dim."""
    return {d: 10 ** (a + b * np.log10(d)) for d in dims}


def make_plot(group_name, best_per_dim, fit_params, extrapolated, plots_dir):
    """Create and save the LR vs hidden_dim plot."""
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all tested LRs, colored by per-dim normalized loss
    # (dark = best at that dim, light = worst at that dim)
    cmap = plt.get_cmap("viridis")
    for dim, (best_lr, best_loss, all_results) in sorted(best_per_dim.items()):
        lrs_tested = np.array([r[0] for r in all_results])
        losses = np.array([r[1] for r in all_results])
        if losses.max() > losses.min():
            norm = (losses - losses.min()) / (losses.max() - losses.min())
        else:
            norm = np.zeros_like(losses)
        colors = cmap(norm)
        sizes = 80 - 50 * norm  # best: 80, worst: 30
        ax.scatter([dim] * len(lrs_tested), lrs_tested,
                   c=colors, alpha=0.8, s=sizes, zorder=1,
                   edgecolors="none",
                   label="Tested LRs (color = per-dim loss, dark=best)" if dim == min(best_per_dim) else None)

    # Plot best LR per dim (bold)
    dims_measured = sorted(best_per_dim.keys())
    best_lrs = [best_per_dim[d][0] for d in dims_measured]
    ax.scatter(dims_measured, best_lrs, c="blue", s=100, zorder=3,
               edgecolors="black", linewidth=1.5, label="Best LR (measured)")

    # Plot fit line
    if fit_params is not None:
        a, b, r2 = fit_params
        x_fit = np.logspace(np.log10(32), np.log10(512), 200)
        y_fit = 10 ** (a + b * np.log10(x_fit))
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.8,
                label=f"Fit: lr = {10**a:.2f} * d^({b:.2f})  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hidden Dimension", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.set_title(f"LR Scaling: {group_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    # Set x-ticks at standard dims
    ax.set_xticks([32, 48, 64, 96, 128, 192, 256, 384, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()

    png_path = os.path.join(plots_dir, f"{group_name}_lr_scaling.png")
    pdf_path = os.path.join(plots_dir, f"{group_name}_lr_scaling.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


def print_results(group_name, best_per_dim, fit_params, extrapolated):
    """Print summary tables to stdout."""
    print(f"\n{'='*70}")
    print(f"  LR SWEEP RESULTS: {group_name}")
    print(f"{'='*70}")

    # Measured best LRs
    print(f"\n  Measured best LR per dimension:")
    print(f"  {'Dim':<8} {'Best LR':<14} {'Val Loss':<12} {'LRs Tested':<10}")
    print(f"  {'-'*44}")
    for dim in sorted(best_per_dim.keys()):
        best_lr, best_loss, all_results = best_per_dim[dim]
        print(f"  {dim:<8} {best_lr:<14.6f} {best_loss:<12.4f} {len(all_results)}")

    if fit_params is None:
        print("\n  Fit failed - not enough data points.")
        return

    a, b, r2 = fit_params
    coeff = 10 ** a
    print(f"\n  Power law fit: lr = {coeff:.4f} * hidden_dim^({b:.4f})")
    print(f"  R² = {r2:.4f}")

    # Extrapolated table
    print(f"\n  Fitted optimal LRs (dims 16-512, step 16):")
    print(f"  {'Dim':<8} {'Fitted LR':<14} {'log10(LR)':<12}")
    print(f"  {'-'*34}")
    for dim in sorted(extrapolated.keys()):
        lr = extrapolated[dim]
        print(f"  {dim:<8} {lr:<14.6f} {np.log10(lr):<12.2f}")

    # Python dict for copy-paste into experiment_definitions.py
    print(f"\n  Copy-paste dict for experiment_definitions.py:")
    print(f"  # {group_name}: lr = {coeff:.4f} * d^({b:.4f}), R²={r2:.3f}")
    items = [f"{d}: {extrapolated[d]:.6f}" for d in sorted(extrapolated.keys())]
    print(f"  OPTIMAL_LRS = {{{', '.join(items)}}}")
    print(f"{'='*70}\n")


def write_optimal_lrs_csv(group_name, extrapolated, fit_params, plots_dir):
    """Write a CSV with the full interpolated/extrapolated LR table."""
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(plots_dir, "optimal_lrs.csv")
    a, b, r2 = fit_params
    with open(csv_path, "w") as f:
        f.write(f"# {group_name}: lr = {10**a:.6f} * hidden_dim^({b:.4f}), R^2={r2:.4f}\n")
        f.write("hidden_dim,learning_rate,log10_lr\n")
        for dim in sorted(extrapolated.keys()):
            lr = extrapolated[dim]
            f.write(f"{dim},{lr:.8f},{np.log10(lr):.4f}\n")
    print(f"  Table saved: {csv_path}")


# ---------------------------------------------------------------------------
# PARABOLIC-VERTEX METHOD (ADDITIVE, NOT A REPLACEMENT)
#
# Per-scale quadratic fit L_val = a*x^2 + b*x + c with x = ln(lr); the fitted
# optimum is eta* = exp(-b/(2a)). The log-eta* values are then fit to
# log10(eta*) = alpha * log10(d) + beta.
#
# This is the procedure used in recent LLM scaling-law work and is kept
# deliberately conservative:
#   - Bjorck et al., "Scaling Optimal LR Across Token Horizons", ICLR 2025:
#     fit a second-degree polynomial in log(LR) using ALL swept points per
#     configuration (typically ~9 after adaptive refinement); remove runs
#     that diverged. R^2 is reported as >= 0.995.
#   - Google VaultGemma technical report (2025): seven LR runs per
#     configuration, fit a quadratic across all seven, take the vertex.
#
# Defaults here mirror that literature: fit all finite points per dim (no
# basin windowing by default); drop only NaN/Inf losses. A `window` argument
# is available if you later want to restrict to the N points around the
# argmin (pass --parabolic-window N, N>=3).
# ---------------------------------------------------------------------------

def fit_quadratic_in_log_lr(lrs, losses, window=None, delta_cut=1.0):
    """
    Fit L = a*x^2 + b*x + c where x = ln(lr).

    By default (window=None), all finite points are used -- matching the
    standard-practice all-points fit in Bjorck et al. 2025 and Google's
    VaultGemma report. If window is an integer >= 3, fit a contiguous window
    of that many points centered on the argmin (truncated at edges).

    A "basin delta-cut" is applied before fitting: points with
    val_loss > L_min + delta_cut (default 1.0 nat) are dropped. These
    correspond to runs that crossed the divergence cliff and converged to
    the uniform-output baseline ~log(vocab) -- a non-NaN plateau that the
    plain finite-mask filter misses. Without this, post-LN / inverse-sqrt
    transformers (Vaswani et al. 2017; Xiong et al. 2020 -- "On Layer
    Normalization in the Transformer Architecture", ICML 2020) produce
    a step-function-shaped loss curve (smooth descent -> cliff -> flat
    plateau) that is not parabolic and yields a biased vertex. The cut
    isolates the basin so a quadratic is the right local model, in the
    spirit of Hägele et al. 2024 ("Scaling Laws and Compute-Optimal
    Training Beyond Fixed Training Durations", NeurIPS 2024) who restrict
    the fit to L <= L_min + delta. Pass delta_cut=None to disable.

    Returns a dict with fit params, vertex eta_star, R^2, residuals, window
    used, and diagnostic flags. Returns None if fewer than 3 finite points.
    """
    lrs = np.asarray(lrs, dtype=float)
    losses = np.asarray(losses, dtype=float)

    # Sort by LR and drop non-finite losses (diverged/crashed runs).
    order = np.argsort(lrs)
    lrs = lrs[order]
    losses = losses[order]
    finite_mask = np.isfinite(losses) & np.isfinite(lrs) & (lrs > 0)
    n_dropped = int((~finite_mask).sum())
    lrs = lrs[finite_mask]
    losses = losses[finite_mask]

    if len(lrs) < 3:
        return None

    # Basin delta-cut: drop divergence-plateau points (loss saturated near
    # log(vocab)). See Xiong et al. 2020 for the post-LN cliff phenomenon
    # and Hägele et al. 2024 for the L <= L_min + delta basin restriction.
    n_dropped_plateau = 0
    if delta_cut is not None and len(lrs) >= 3:
        L_min = float(np.min(losses))
        keep = losses <= L_min + float(delta_cut)
        n_dropped_plateau = int((~keep).sum())
        if int(keep.sum()) >= 3:
            lrs = lrs[keep]
            losses = losses[keep]
        else:
            n_dropped_plateau = 0  # not enough basin points; keep all

    # Optional basin window around the argmin.
    if window is not None and window >= 3 and len(lrs) > window:
        best_idx = int(np.argmin(losses))
        half = window // 2
        lo = max(0, best_idx - half)
        hi = min(len(lrs), lo + window)
        lo = max(0, hi - window)  # shift left if we hit the right edge
        sel_lrs = lrs[lo:hi]
        sel_losses = losses[lo:hi]
        window_used = window
    else:
        sel_lrs = lrs
        sel_losses = losses
        window_used = len(lrs)

    x = np.log(sel_lrs)
    coeffs = np.polyfit(x, sel_losses, 2)  # [a, b, c]
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    pred = np.polyval(coeffs, x)
    residuals = (sel_losses - pred).tolist()
    ss_res = float(np.sum((sel_losses - pred) ** 2))
    ss_tot = float(np.sum((sel_losses - np.mean(sel_losses)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    warnings_ = []
    unstable = False
    if a <= 0:
        unstable = True
        warnings_.append("parabola not convex (a<=0); no interior minimum")
        eta_star = float("nan")
        x_star = float("nan")
    else:
        x_star = -b / (2.0 * a)
        eta_star = float(np.exp(x_star))
        swept_lo = float(lrs[0])
        swept_hi = float(lrs[-1])
        if eta_star < swept_lo or eta_star > swept_hi:
            warnings_.append(
                f"vertex eta*={eta_star:.3e} outside swept range [{swept_lo:.3e}, {swept_hi:.3e}]"
            )
    if n_dropped:
        warnings_.append(f"dropped {n_dropped} non-finite point(s)")
    if n_dropped_plateau:
        warnings_.append(f"dropped {n_dropped_plateau} divergence-plateau point(s)")

    return {
        "eta_star": eta_star,
        "x_star": x_star,
        "a": a, "b": b, "c": c,
        "r2": r2,
        "residuals": residuals,
        "window_used": window_used,
        "window_lrs": sel_lrs.tolist(),
        "window_losses": sel_losses.tolist(),
        "all_lrs": lrs.tolist(),
        "all_losses": losses.tolist(),
        "best_grid_lr": float(lrs[int(np.argmin(losses))]),
        "unstable": unstable,
        "n_dropped": n_dropped,
        "n_dropped_plateau": n_dropped_plateau,
        "warnings": warnings_,
    }


def fit_parabolic_per_dim(best_per_dim, window=None):
    """Per-dim quadratic-in-log-lr fit. Returns {dim: fit_dict}."""
    out = {}
    for dim, (_best_lr, _best_loss, results) in sorted(best_per_dim.items()):
        lrs = [r[0] for r in results]
        losses = [r[1] for r in results]
        fit = fit_quadratic_in_log_lr(lrs, losses, window=window)
        if fit is not None:
            out[dim] = fit
    return out


def _vertex_in_swept_range(fit, edge_margin_decades=0.1):
    """
    True iff the vertex sits at least `edge_margin_decades` decades inside
    both swept-LR boundaries. The margin guards against the failure mode
    where the loss curve is still descending at the swept upper (or lower)
    edge: a quadratic happily fits the descending side and reports R^2 near
    1.0, but the vertex pegs to the boundary and the inferred optimum is
    actually outside the evidence. Same protocol used in Bjorck et al. 2025
    ("Scaling Optimal Learning Rate Across Token Horizons", ICLR 2025), who
    iteratively refine the LR grid until the vertex sits in the interior.
    Set edge_margin_decades=0 to recover strict in-range behavior.
    """
    if not (np.isfinite(fit["eta_star"]) and fit["all_lrs"]):
        return False
    lr_lo = float(fit["all_lrs"][0])
    lr_hi = float(fit["all_lrs"][-1])
    log_eta = np.log10(fit["eta_star"])
    return bool(
        log_eta - np.log10(lr_lo) >= edge_margin_decades
        and np.log10(lr_hi) - log_eta >= edge_margin_decades
    )


def _parabolic_vertex_valid(fit, r2_min=0.9):
    """
    Validity gates for using the parabolic vertex as the per-dim eta* estimate:
      (1) parabola convex (a > 0), vertex finite        [unstable test]
      (2) vertex sits >= 0.1 decades inside swept range  [boundary test, Bjorck 2025]
      (3) parabola R^2 >= r2_min on the fit window       [shape test]
    A failure on any gate means the loss curve is not well-approximated by a
    quadratic in this region (cliff, plateau, or sweep didn't bracket the
    optimum), so we should fall back to the grid argmin for this dim.
    """
    if fit["unstable"] or not np.isfinite(fit["eta_star"]):
        return False
    if not _vertex_in_swept_range(fit):
        return False
    if not np.isfinite(fit["r2"]) or fit["r2"] < r2_min:
        return False
    return True


def fit_lr_scaling_parabolic(parabolic_per_dim, r2_min=0.9):
    """
    Fit log10(eta*) = alpha * log10(d) + beta across dims.

    Per-dim estimator selection (cross-arch fairness protocol):
      - Use the parabolic vertex eta* when validity gates pass
        (Bjorck et al. 2025; VaultGemma 2025 -- vertex of L_val(ln lr)).
      - Otherwise fall back to the grid argmin best_grid_lr for that dim.
        This is what Hoffmann et al. 2022 (Chinchilla) and Kaplan et al. 2020
        report; it stays robust on architectures whose loss curve is not
        locally quadratic (e.g. post-LN transformers with a divergence cliff,
        Vaswani 2017 / Xiong 2020).

    Reporting both estimators side-by-side is recommended by Porian et al.
    2024 ("Resolving Discrepancies in Compute-Optimal Scaling of Language
    Models"), who attribute much of the Chinchilla-vs-Kaplan slope
    discrepancy to differing LR-tune protocols.

    Returns the fit plus a per-dim record of which estimator was used.
    """
    per_dim_estimator = {}   # d -> ("vertex" | "argmin", eta_used, reason)
    for d, f in parabolic_per_dim.items():
        if _parabolic_vertex_valid(f, r2_min=r2_min):
            per_dim_estimator[d] = ("vertex", float(f["eta_star"]), "")
        else:
            reasons = []
            if f["unstable"]:
                reasons.append("a<=0")
            if not np.isfinite(f["eta_star"]):
                reasons.append("nan_vertex")
            if np.isfinite(f["eta_star"]) and not _vertex_in_swept_range(f):
                reasons.append("vertex_at_boundary")
            if np.isfinite(f["r2"]) and f["r2"] < r2_min:
                reasons.append(f"r2<{r2_min}")
            per_dim_estimator[d] = (
                "argmin", float(f["best_grid_lr"]), ",".join(reasons) or "fallback"
            )

    if len(per_dim_estimator) < 2:
        return None
    dims_sorted = sorted(per_dim_estimator.keys())
    dims = np.array(dims_sorted, dtype=float)
    lrs = np.array([per_dim_estimator[int(d)][1] for d in dims])
    log_dims = np.log10(dims)
    log_lrs = np.log10(lrs)
    slope, intercept = np.polyfit(log_dims, log_lrs, 1)
    pred = slope * log_dims + intercept
    ss_res = float(np.sum((log_lrs - pred) ** 2))
    ss_tot = float(np.sum((log_lrs - np.mean(log_lrs)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    vertex_dims = sorted(d for d, v in per_dim_estimator.items() if v[0] == "vertex")
    argmin_dims = sorted(d for d, v in per_dim_estimator.items() if v[0] == "argmin")
    return {
        "alpha": float(slope),
        "beta": float(intercept),
        "r2": float(r2),
        "usable_dims": dims_sorted,
        "excluded_dims": [],
        "per_dim_estimator": per_dim_estimator,
        "vertex_dims": vertex_dims,
        "argmin_fallback_dims": argmin_dims,
        "r2_min": float(r2_min),
    }


def make_parabolic_per_dim_plot(group_name, parabolic_per_dim, plots_dir):
    """Grid of per-scale sweeps with the quadratic fit and vertex overlaid."""
    if not parabolic_per_dim:
        return
    os.makedirs(plots_dir, exist_ok=True)
    dims = sorted(parabolic_per_dim.keys())
    n = len(dims)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)

    for ax, dim in zip(axes.flatten(), dims):
        fit = parabolic_per_dim[dim]
        all_lrs = np.array(fit["all_lrs"])
        all_losses = np.array(fit["all_losses"])
        win_lrs = np.array(fit["window_lrs"])
        win_losses = np.array(fit["window_losses"])
        # Faded background = full sweep; highlighted = window actually fit
        ax.scatter(all_lrs, all_losses, c="lightgray", s=40, label="swept")
        if len(win_lrs) != len(all_lrs):
            ax.scatter(win_lrs, win_losses, c="tab:blue", s=60,
                       edgecolors="black", linewidth=0.8, label="fit window")
        else:
            ax.scatter(win_lrs, win_losses, c="tab:blue", s=60,
                       edgecolors="black", linewidth=0.8, label="fit points")
        if not fit["unstable"]:
            span_lo = np.log(min(win_lrs.min(), fit["eta_star"]) * 0.9)
            span_hi = np.log(max(win_lrs.max(), fit["eta_star"]) * 1.1)
            x_grid = np.linspace(span_lo, span_hi, 200)
            y_grid = fit["a"] * x_grid ** 2 + fit["b"] * x_grid + fit["c"]
            ax.plot(np.exp(x_grid), y_grid, "r-", linewidth=2, alpha=0.85,
                    label=f"quadratic (R²={fit['r2']:.3f})")
            ax.axvline(fit["eta_star"], color="red", linestyle="--", alpha=0.7,
                       label=f"η*={fit['eta_star']:.2e}")
        title_bits = [f"d={dim}"]
        if fit["unstable"]:
            title_bits.append("UNSTABLE")
        elif fit["warnings"]:
            title_bits.append("warn")
        ax.set_title(" — ".join(title_bits), fontsize=11)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("final val loss")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")

    for ax in axes.flatten()[len(dims):]:
        ax.axis("off")

    fig.suptitle(
        f"Parabolic LR fits per scale: {group_name}\n"
        f"(quadratic in ln(lr); vertex η* = exp(-b/(2a)); standard per Bjorck 2025 / VaultGemma)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(plots_dir, f"{group_name}_parabolic_per_dim.png")
    pdf_path = os.path.join(plots_dir, f"{group_name}_parabolic_per_dim.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


def make_parabolic_scaling_plot(group_name, parabolic_per_dim, parabolic_fit, plots_dir):
    """log10(eta*) vs log10(d), color-coded by per-dim estimator."""
    if parabolic_fit is None:
        return
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    per_dim_est = parabolic_fit.get("per_dim_estimator", {})
    plotted_vertex = plotted_argmin = False
    for d in sorted(parabolic_per_dim.keys()):
        if d not in per_dim_est:
            continue
        est_name, eta_used, _ = per_dim_est[d]
        if est_name == "vertex":
            ax.scatter(d, eta_used, c="tab:blue", s=120,
                       edgecolors="black", linewidth=1.2, zorder=3,
                       label="η* (parabolic vertex)" if not plotted_vertex else None)
            plotted_vertex = True
        else:
            ax.scatter(d, eta_used, c="tab:orange", marker="s", s=120,
                       edgecolors="black", linewidth=1.2, zorder=3,
                       label="η* (argmin fallback)" if not plotted_argmin else None)
            plotted_argmin = True

    alpha = parabolic_fit["alpha"]
    beta = parabolic_fit["beta"]
    r2 = parabolic_fit["r2"]
    lo_d = min(parabolic_fit["usable_dims"])
    hi_d = max(parabolic_fit["usable_dims"])
    x_fit = np.logspace(np.log10(lo_d), np.log10(hi_d), 200)
    y_fit = 10 ** (alpha * np.log10(x_fit) + beta)
    ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.85,
            label=f"η* = {10**beta:.3e} · d^({alpha:.3f})  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hidden Dimension", fontsize=14)
    ax.set_ylabel("η*  (parabolic vertex)", fontsize=14)
    ax.set_title(f"Parabolic-vertex LR scaling: {group_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([32, 48, 64, 96, 128, 192, 256, 384, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    png_path = os.path.join(plots_dir, f"{group_name}_parabolic_scaling.png")
    pdf_path = os.path.join(plots_dir, f"{group_name}_parabolic_scaling.pdf")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {png_path}")
    print(f"  Plot saved: {pdf_path}")


def write_parabolic_csvs(group_name, parabolic_per_dim, parabolic_fit, target_dims, plots_dir):
    """
    Write two CSVs for the parabolic method (separate from the default outputs):
      - parabolic_per_dim.csv: per-scale diagnostics (vertex eta*, R^2, flags)
      - optimal_lrs_parabolic.csv: extrapolation table (dims 32..512 step 16)
    """
    os.makedirs(plots_dir, exist_ok=True)
    per_dim_est = parabolic_fit.get("per_dim_estimator", {}) if parabolic_fit else {}
    diag_path = os.path.join(plots_dir, "parabolic_per_dim.csv")
    with open(diag_path, "w") as f:
        f.write(
            "hidden_dim,estimator,eta_used,reason,eta_star_parabolic,best_grid_lr,"
            "r_squared,a,b,c,window_used,unstable,vertex_in_swept_range,"
            "n_dropped,n_dropped_plateau,warnings\n"
        )
        for d in sorted(parabolic_per_dim.keys()):
            fit = parabolic_per_dim[d]
            in_range = bool(
                np.isfinite(fit["eta_star"])
                and fit["all_lrs"][0] <= fit["eta_star"] <= fit["all_lrs"][-1]
            )
            warn_str = "; ".join(fit["warnings"]).replace(",", ";").replace("\n", " ")
            est_name, eta_used, reason = per_dim_est.get(d, ("n/a", float("nan"), ""))
            f.write(
                f"{d},{est_name},{eta_used:.8g},\"{reason}\","
                f"{fit['eta_star']:.8g},{fit['best_grid_lr']:.8g},"
                f"{fit['r2']:.6f},{fit['a']:.6g},{fit['b']:.6g},{fit['c']:.6g},"
                f"{fit['window_used']},{fit['unstable']},{in_range},"
                f"{fit['n_dropped']},{fit.get('n_dropped_plateau', 0)},\"{warn_str}\"\n"
            )
    print(f"  Table saved: {diag_path}")

    if parabolic_fit is None:
        return
    alpha = parabolic_fit["alpha"]
    beta = parabolic_fit["beta"]
    r2 = parabolic_fit["r2"]
    table_path = os.path.join(plots_dir, "optimal_lrs_parabolic.csv")
    with open(table_path, "w") as f:
        f.write(
            f"# {group_name}: parabolic-vertex scaling law, "
            f"eta* = {10**beta:.6f} * d^({alpha:.4f}), R^2={r2:.4f}\n"
        )
        f.write(
            "# Method: per-scale quadratic L_val = a*x^2 + b*x + c, x=ln(lr); "
            "eta* = exp(-b/(2a)); then log10(eta*) = alpha*log10(d) + beta.\n"
        )
        f.write("# Standard procedure in Bjorck et al. 2025 (ICLR) and Google VaultGemma report.\n")
        f.write("hidden_dim,learning_rate,log10_lr\n")
        for d in target_dims:
            lr = 10 ** (alpha * np.log10(d) + beta)
            f.write(f"{d},{lr:.8f},{np.log10(lr):.4f}\n")
    print(f"  Table saved: {table_path}")


def print_parabolic_results(group_name, parabolic_per_dim, parabolic_fit, window_arg):
    print(f"\n  ---- PARABOLIC-VERTEX (with argmin fallback) : {group_name} ----")
    if window_arg is None:
        print(f"  Window: all finite points per scale (Bjorck 2025, VaultGemma).")
    else:
        print(f"  Window: best {window_arg} contiguous points around per-scale argmin.")
    print(f"  Per-dim estimator: vertex if validity gates pass, else grid-argmin")
    print(f"    (Chinchilla-style fallback; cross-arch fairness, Porian et al. 2024).")
    print(f"  Fit: L_val = a*x^2 + b*x + c in x=ln(lr); eta* = exp(-b/(2a)).")
    print(f"\n  {'Dim':<6} {'estimator':<10} {'eta_used':<13} {'vertex':<13} {'argmin':<13} {'R²':<7} {'n':<4} {'flag'}")
    print(f"  {'-'*92}")
    per_dim_est = parabolic_fit["per_dim_estimator"] if parabolic_fit else {}
    for d in sorted(parabolic_per_dim.keys()):
        fit = parabolic_per_dim[d]
        flag_bits = []
        if fit["unstable"]:
            flag_bits.append("UNSTABLE")
        elif np.isfinite(fit["eta_star"]) and not (fit["all_lrs"][0] <= fit["eta_star"] <= fit["all_lrs"][-1]):
            flag_bits.append("EXTRAPOLATED")
        if fit["n_dropped"]:
            flag_bits.append(f"{fit['n_dropped']}dropped")
        if fit.get("n_dropped_plateau"):
            flag_bits.append(f"{fit['n_dropped_plateau']}plateau")
        flag = ",".join(flag_bits) or "ok"
        vertex_str = f"{fit['eta_star']:.3e}" if np.isfinite(fit["eta_star"]) else "NaN"
        argmin_str = f"{fit['best_grid_lr']:.3e}"
        if d in per_dim_est:
            est_name, eta_used, reason = per_dim_est[d]
            est_label = est_name if not reason else f"{est_name}({reason})"
            eta_str = f"{eta_used:.3e}"
        else:
            est_label = "n/a"
            eta_str = "n/a"
        print(
            f"  {d:<6} {est_label:<10} {eta_str:<13} {vertex_str:<13} {argmin_str:<13} "
            f"{fit['r2']:<7.3f} {fit['window_used']:<4} {flag}"
        )

    if parabolic_fit is None:
        print("\n  Scaling fit: skipped (fewer than 2 usable dims).")
        return
    alpha = parabolic_fit["alpha"]
    beta = parabolic_fit["beta"]
    r2 = parabolic_fit["r2"]
    print(f"\n  Cross-arch scaling law: log10(η*) = {alpha:.4f} · log10(d) + {beta:.4f}")
    print(f"  Equivalently: η* = {10**beta:.6f} · d^({alpha:.4f}),  R²={r2:.4f}")
    n_v = len(parabolic_fit["vertex_dims"])
    n_a = len(parabolic_fit["argmin_fallback_dims"])
    print(f"  Estimator mix: vertex on {n_v} dim(s) {parabolic_fit['vertex_dims']}, "
          f"argmin fallback on {n_a} dim(s) {parabolic_fit['argmin_fallback_dims']}")


def analyze_group(group_name, include_parabolic=True, parabolic_window=None):
    """Run the full analysis pipeline for one group (concrete or combined)."""
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots", group_name)

    # Resolve to one or more concrete results dirs.
    if group_name in LR_EXPERIMENT_GROUPS and is_combined_group(group_name):
        sub_names = expand_subgroups(group_name)
        results_dirs = [os.path.join(base_dir, "results", s) for s in sub_names]
        print(f"\nAnalyzing combined group: {group_name}")
        print(f"  Subgroups: {sub_names}")
        for d in results_dirs:
            print(f"    Results dir: {d}")
    else:
        results_dirs = [os.path.join(base_dir, "results", group_name)]
        print(f"\nAnalyzing group: {group_name}")
        print(f"  Results dir: {results_dirs[0]}")

    # Step 1: Find best LR per dimension (pools across results dirs for combined groups)
    best_per_dim = find_best_lr_per_dim(results_dirs)
    if not best_per_dim:
        print(f"  No results found for group '{group_name}'. Skipping.")
        return

    # Step 2: Fit power law
    fit_params = fit_lr_scaling(best_per_dim)

    # Step 3: Extrapolate/interpolate to all target dims (32 to 512, step 16)
    target_dims = list(range(32, 513, 16))
    extrapolated = extrapolate_lrs(fit_params[0], fit_params[1], target_dims) if fit_params else {}

    # Step 4: Plot (without the extrapolated table overlaid)
    make_plot(group_name, best_per_dim, fit_params, extrapolated, plots_dir)

    # Step 5: Write the full LR table to CSV (separate from the plot)
    if fit_params is not None:
        write_optimal_lrs_csv(group_name, extrapolated, fit_params, plots_dir)

    # Step 6: Print tables to stdout
    print_results(group_name, best_per_dim, fit_params, extrapolated)

    # Step 7 (ADDITIVE): Parabolic-vertex method alongside the default.
    # The default outputs above are untouched; this writes extra files with a
    # `_parabolic` suffix and prints a second diagnostic block.
    if include_parabolic:
        parabolic_per_dim = fit_parabolic_per_dim(best_per_dim, window=parabolic_window)
        if not parabolic_per_dim:
            print("  [parabolic] Not enough finite points per dim to fit any quadratic; skipping.")
            return
        parabolic_fit = fit_lr_scaling_parabolic(parabolic_per_dim)
        make_parabolic_per_dim_plot(group_name, parabolic_per_dim, plots_dir)
        if parabolic_fit is not None:
            make_parabolic_scaling_plot(group_name, parabolic_per_dim, parabolic_fit, plots_dir)
        write_parabolic_csvs(group_name, parabolic_per_dim, parabolic_fit, target_dims, plots_dir)
        print_parabolic_results(group_name, parabolic_per_dim, parabolic_fit, parabolic_window)

        # N-axis fit (Kaplan/Chinchilla/Bjorck convention) keyed back to hidden_dim.
        group_config = _resolve_concrete_group(group_name)
        fit_N = fit_lr_scaling_vs_N(parabolic_fit, group_config)
        make_parabolic_scaling_vs_N_plot(group_name, parabolic_fit, fit_N, group_config, plots_dir)
        make_d_vs_N_comparison_plot(group_name, parabolic_fit, fit_N, group_config, plots_dir)
        write_parabolic_vs_N_csv(group_name, fit_N, group_config, target_dims, plots_dir)
        print_parabolic_vs_N_results(group_name, fit_N, group_config, target_dims)


def main():
    parser = argparse.ArgumentParser(description="Analyze LR sweep results")
    parser.add_argument("--group", type=str, default=None,
                        help="Name of the LR experiment group to analyze")
    parser.add_argument("--all-groups", action="store_true",
                        help="Analyze all groups that have results")
    parser.add_argument("--no-parabolic", action="store_true",
                        help="Skip the parabolic-vertex LR analysis "
                             "(by default it runs alongside the grid-argmin fit).")
    parser.add_argument("--parabolic-window", type=int, default=None,
                        help="If set, fit the quadratic to this many contiguous "
                             "points around the per-scale argmin (minimum 3). "
                             "Default: use all finite swept points, matching "
                             "standard practice (Bjorck 2025 / VaultGemma).")
    args = parser.parse_args()

    if args.parabolic_window is not None and args.parabolic_window < 3:
        print("--parabolic-window must be >= 3 (need at least 3 points for a quadratic).")
        sys.exit(1)

    include_parabolic = not args.no_parabolic

    if args.all_groups:
        base_dir = os.path.dirname(__file__)
        results_base = os.path.join(base_dir, "results")
        if os.path.isdir(results_base):
            groups = [d for d in os.listdir(results_base)
                      if os.path.isdir(os.path.join(results_base, d))]
            for g in sorted(groups):
                analyze_group(g, include_parabolic=include_parabolic,
                              parabolic_window=args.parabolic_window)
        else:
            print("No results directory found.")
    elif args.group:
        if args.group not in LR_EXPERIMENT_GROUPS:
            print(f"Warning: '{args.group}' not in LR_EXPERIMENT_GROUPS, "
                  f"but will try to analyze results anyway.")
        analyze_group(args.group, include_parabolic=include_parabolic,
                      parabolic_window=args.parabolic_window)
    else:
        print("Specify --group <name> or --all-groups")
        sys.exit(1)


if __name__ == "__main__":
    main()
