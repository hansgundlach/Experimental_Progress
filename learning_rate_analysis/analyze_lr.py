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

def fit_quadratic_in_log_lr(lrs, losses, window=None):
    """
    Fit L = a*x^2 + b*x + c where x = ln(lr).

    By default (window=None), all finite points are used -- matching the
    standard-practice all-points fit in Bjorck et al. 2025 and Google's
    VaultGemma report. If window is an integer >= 3, fit a contiguous window
    of that many points centered on the argmin (truncated at edges).

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


def _vertex_in_swept_range(fit):
    return bool(
        np.isfinite(fit["eta_star"])
        and fit["all_lrs"][0] <= fit["eta_star"] <= fit["all_lrs"][-1]
    )


def fit_lr_scaling_parabolic(parabolic_per_dim):
    """
    Fit log10(eta*) = alpha * log10(d) + beta using vertex-estimated eta*.
    Dims are excluded from the scaling fit if the parabola was unstable (a<=0),
    the vertex is non-finite, or the vertex lies outside the swept LR range
    (in which case the quadratic has extrapolated past the evidence and the
    "optimum" is not supported by the sweep -- the same logic that motivates
    the per-dim warning).
    """
    usable = {
        d: f["eta_star"]
        for d, f in parabolic_per_dim.items()
        if (not f["unstable"])
        and np.isfinite(f["eta_star"])
        and _vertex_in_swept_range(f)
    }
    if len(usable) < 2:
        return None
    dims = np.array(sorted(usable.keys()), dtype=float)
    lrs = np.array([usable[int(d)] for d in dims])
    log_dims = np.log10(dims)
    log_lrs = np.log10(lrs)
    slope, intercept = np.polyfit(log_dims, log_lrs, 1)
    pred = slope * log_dims + intercept
    ss_res = float(np.sum((log_lrs - pred) ** 2))
    ss_tot = float(np.sum((log_lrs - np.mean(log_lrs)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "alpha": float(slope),
        "beta": float(intercept),
        "r2": float(r2),
        "usable_dims": sorted(usable.keys()),
        "excluded_dims": sorted(d for d in parabolic_per_dim if d not in usable),
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
    """log10(eta*) vs log10(d), using parabolic-vertex LRs as the data."""
    if parabolic_fit is None:
        return
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    usable = set(parabolic_fit["usable_dims"])
    for d, f in sorted(parabolic_per_dim.items()):
        if not np.isfinite(f["eta_star"]):
            continue
        is_usable = d in usable
        ax.scatter(
            d, f["eta_star"],
            c="tab:blue" if is_usable else "lightgray",
            s=120,
            edgecolors="black" if is_usable else "none",
            linewidth=1.2, zorder=3,
            label=("η* (vertex, used in fit)" if is_usable and d == min(usable) else None)
                  if is_usable else
                  ("η* (vertex, excluded: unstable or out-of-range)"
                   if (not is_usable) and d == min(parabolic_per_dim.keys() - usable, default=None) else None),
        )

    alpha = parabolic_fit["alpha"]
    beta = parabolic_fit["beta"]
    r2 = parabolic_fit["r2"]
    lo_d, hi_d = min(usable), max(usable)
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
    diag_path = os.path.join(plots_dir, "parabolic_per_dim.csv")
    with open(diag_path, "w") as f:
        f.write(
            "hidden_dim,eta_star_parabolic,best_grid_lr,r_squared,a,b,c,"
            "window_used,unstable,vertex_in_swept_range,n_dropped,warnings\n"
        )
        for d in sorted(parabolic_per_dim.keys()):
            fit = parabolic_per_dim[d]
            in_range = bool(
                np.isfinite(fit["eta_star"])
                and fit["all_lrs"][0] <= fit["eta_star"] <= fit["all_lrs"][-1]
            )
            warn_str = "; ".join(fit["warnings"]).replace(",", ";").replace("\n", " ")
            f.write(
                f"{d},{fit['eta_star']:.8g},{fit['best_grid_lr']:.8g},"
                f"{fit['r2']:.6f},{fit['a']:.6g},{fit['b']:.6g},{fit['c']:.6g},"
                f"{fit['window_used']},{fit['unstable']},{in_range},"
                f"{fit['n_dropped']},\"{warn_str}\"\n"
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
    print(f"\n  ---- PARABOLIC-VERTEX METHOD : {group_name} ----")
    if window_arg is None:
        print(f"  Window: all finite points per scale (standard; matches Bjorck 2025, VaultGemma).")
    else:
        print(f"  Window: best {window_arg} contiguous points around per-scale argmin.")
    print(f"  Fit: L_val = a*x^2 + b*x + c in x=ln(lr); eta* = exp(-b/(2a)).")
    print(f"\n  {'Dim':<6} {'η* (vertex)':<14} {'best grid η':<14} {'R²':<8} {'n':<4} {'flag'}")
    print(f"  {'-'*68}")
    for d in sorted(parabolic_per_dim.keys()):
        fit = parabolic_per_dim[d]
        flag_bits = []
        if fit["unstable"]:
            flag_bits.append("UNSTABLE")
        elif np.isfinite(fit["eta_star"]) and not (fit["all_lrs"][0] <= fit["eta_star"] <= fit["all_lrs"][-1]):
            flag_bits.append("EXTRAPOLATED")
        if fit["n_dropped"]:
            flag_bits.append(f"{fit['n_dropped']}dropped")
        flag = ",".join(flag_bits) or "ok"
        eta_str = f"{fit['eta_star']:.4e}" if np.isfinite(fit["eta_star"]) else "NaN"
        print(
            f"  {d:<6} {eta_str:<14} {fit['best_grid_lr']:<14.4e} "
            f"{fit['r2']:<8.3f} {fit['window_used']:<4} {flag}"
        )

    if parabolic_fit is None:
        print("\n  Scaling fit: skipped (fewer than 2 usable dims).")
        return
    alpha = parabolic_fit["alpha"]
    beta = parabolic_fit["beta"]
    r2 = parabolic_fit["r2"]
    print(f"\n  Parabolic scaling law: log10(η*) = {alpha:.4f} · log10(d) + {beta:.4f}")
    print(f"  Equivalently: η* = {10**beta:.6f} · d^({alpha:.4f}),  R²={r2:.4f}")
    if parabolic_fit["excluded_dims"]:
        print(f"  Excluded from scaling fit (unstable/NaN vertex): {parabolic_fit['excluded_dims']}")


def analyze_group(group_name, include_parabolic=True, parabolic_window=None):
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
