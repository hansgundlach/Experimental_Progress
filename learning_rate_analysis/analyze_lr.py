"""
Analyze LR sweep results: find best LR per dimension, fit scaling law, plot, and
output a table of fitted optimal LRs for hidden_dims 16-512 (step 16).

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
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir for imports if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lr_experiment_groups import LR_EXPERIMENT_GROUPS


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

    try:
        # Handle formats like "10e+0", "10e-1", "3e-02", "1.8e-02"
        lr = float(lr_str.replace("e", "e"))
        return dim, lr
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


def find_best_lr_per_dim(results_dir):
    """
    Scan all CSVs in results_dir and return {dim: (best_lr, best_loss, all_results)}.
    all_results is a list of (lr, loss) for that dim.
    """
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return {}

    csv_files = sorted(Path(results_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return {}

    # Collect all (dim, lr, loss) triples
    records = []
    for csv_path in csv_files:
        dim, lr = parse_lr_csv_name(csv_path.name)
        if dim is None or lr is None:
            continue
        loss = get_final_val_loss(str(csv_path))
        if loss is None or np.isnan(loss):
            continue
        records.append((dim, lr, loss))

    if not records:
        print(f"No valid results found in {results_dir}")
        return {}

    # Group by dimension
    from collections import defaultdict
    by_dim = defaultdict(list)
    for dim, lr, loss in records:
        by_dim[dim].append((lr, loss))

    # Find best LR per dimension
    best = {}
    for dim, results in sorted(by_dim.items()):
        results.sort(key=lambda x: x[1])  # sort by loss
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

    # Plot all tested LRs (faded)
    for dim, (best_lr, best_loss, all_results) in sorted(best_per_dim.items()):
        lrs_tested = [r[0] for r in all_results]
        losses = [r[1] for r in all_results]
        ax.scatter([dim] * len(lrs_tested), lrs_tested,
                   c="gray", alpha=0.3, s=20, zorder=1)

    # Plot best LR per dim (bold)
    dims_measured = sorted(best_per_dim.keys())
    best_lrs = [best_per_dim[d][0] for d in dims_measured]
    ax.scatter(dims_measured, best_lrs, c="blue", s=100, zorder=3,
               edgecolors="black", linewidth=1.5, label="Best LR (measured)")

    # Plot fit line
    if fit_params is not None:
        a, b, r2 = fit_params
        x_fit = np.logspace(np.log10(16), np.log10(512), 200)
        y_fit = 10 ** (a + b * np.log10(x_fit))
        ax.plot(x_fit, y_fit, "r--", linewidth=2, alpha=0.8,
                label=f"Fit: lr = {10**a:.2f} * d^({b:.2f})  R²={r2:.3f}")

    # Plot extrapolated points
    if extrapolated:
        ext_dims = sorted(extrapolated.keys())
        ext_lrs = [extrapolated[d] for d in ext_dims]
        ax.scatter(ext_dims, ext_lrs, c="red", s=30, marker="x", zorder=2,
                   alpha=0.5, label="Extrapolated LRs")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hidden Dimension", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.set_title(f"LR Scaling: {group_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    # Set x-ticks at standard dims
    ax.set_xticks([16, 32, 48, 64, 96, 128, 192, 256, 384, 512])
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


def analyze_group(group_name):
    """Run the full analysis pipeline for one group."""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results", group_name)
    plots_dir = os.path.join(base_dir, "plots", group_name)

    print(f"\nAnalyzing group: {group_name}")
    print(f"  Results dir: {results_dir}")

    # Step 1: Find best LR per dimension
    best_per_dim = find_best_lr_per_dim(results_dir)
    if not best_per_dim:
        print(f"  No results found for group '{group_name}'. Skipping.")
        return

    # Step 2: Fit power law
    fit_params = fit_lr_scaling(best_per_dim)

    # Step 3: Extrapolate to all target dims (16 to 512, step 16)
    target_dims = list(range(16, 513, 16))
    extrapolated = extrapolate_lrs(fit_params[0], fit_params[1], target_dims) if fit_params else {}

    # Step 4: Plot
    make_plot(group_name, best_per_dim, fit_params, extrapolated, plots_dir)

    # Step 5: Print tables
    print_results(group_name, best_per_dim, fit_params, extrapolated)


def main():
    parser = argparse.ArgumentParser(description="Analyze LR sweep results")
    parser.add_argument("--group", type=str, default=None,
                        help="Name of the LR experiment group to analyze")
    parser.add_argument("--all-groups", action="store_true",
                        help="Analyze all groups that have results")
    args = parser.parse_args()

    if args.all_groups:
        base_dir = os.path.dirname(__file__)
        results_base = os.path.join(base_dir, "results")
        if os.path.isdir(results_base):
            groups = [d for d in os.listdir(results_base)
                      if os.path.isdir(os.path.join(results_base, d))]
            for g in sorted(groups):
                analyze_group(g)
        else:
            print("No results directory found.")
    elif args.group:
        if args.group not in LR_EXPERIMENT_GROUPS:
            print(f"Warning: '{args.group}' not in LR_EXPERIMENT_GROUPS, "
                  f"but will try to analyze results anyway.")
        analyze_group(args.group)
    else:
        print("Specify --group <name> or --all-groups")
        sys.exit(1)


if __name__ == "__main__":
    main()
