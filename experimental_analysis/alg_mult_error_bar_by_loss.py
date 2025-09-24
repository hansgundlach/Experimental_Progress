# %%
#!/usr/bin/env python3
"""
Compute efficiency multipliers using the by-loss method and recreate the
bar chart from alg_mult_error_bar.py with the new metric.

This script mirrors the comparisons/plotting of alg_mult_error_bar.py, but
each multiplier is computed as the ratio of compute needed to reach the same
target validation loss (Model B / Model A), aggregated across matching seeds
when available.
"""
# %%
import os
import re
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# Robust loader to access compute_multiplier_by_loss without static imports
from importlib import util as _importlib_util


def _get_compute_multiplier_by_loss_func():
    try:
        # Try importing as package module
        import importlib

        module = importlib.import_module(
            "experimental_analysis.compute_multiplier_by_loss"
        )
        return getattr(module, "compute_multiplier_by_loss")
    except Exception:
        # Load directly from file path (same directory)
        candidate = os.path.join(
            os.path.dirname(__file__), "compute_multiplier_by_loss.py"
        )
        spec = _importlib_util.spec_from_file_location(
            "compute_multiplier_by_loss", candidate
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                "Unable to locate compute_multiplier_by_loss.py in experimental_analysis/"
            )
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return getattr(module, "compute_multiplier_by_loss")


# Bind the function name used below
compute_multiplier_by_loss = _get_compute_multiplier_by_loss_func()
# %%

# ===== EASILY ADJUSTABLE PARAMETERS =====
TARGET_LOSS = 5.3  # Target validation loss at which to compare compute
LOSS_COLUMN = "validation_loss"
CONFIDENCE_INTERVALS = False  # Set to True to display error bars

# Plot font sizes (kept consistent with alg_mult_error_bar.py)
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 14
X_TICK_LABEL_FONTSIZE = 12
Y_TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 15
VALUE_LABEL_FONTSIZE = 12
X_TICK_ROTATION = 45


def _get_base_data_path() -> str:
    """Return base path to experimental data, compatible with both run locations."""
    if os.path.exists("../experimental_data_folder/"):
        return "../experimental_data_folder/"
    return "experimental_data_folder/"


def _collect_seeded_files(file_prefix: str) -> Dict[Optional[int], List[str]]:
    """
    Collect CSV files for a given prefix and group them by numeric seed suffix.

    A file is considered seed-tagged if it ends with _<digits>.csv. Files without
    such a suffix are grouped under key None.
    """
    base_path = _get_base_data_path()
    pattern = f"{base_path}{file_prefix}*.csv"
    all_files = glob.glob(pattern)

    seed_regex = re.compile(r"_(\d+)\.csv$")
    grouped: Dict[Optional[int], List[str]] = {}

    for path in all_files:
        match = seed_regex.search(path)
        if match:
            seed = int(match.group(1))
        else:
            seed = None
        grouped.setdefault(seed, []).append(path)

    return grouped


def _pair_seed_files(prefix_a: str, prefix_b: str) -> List[Tuple[str, str]]:
    """
    Pair files from two prefixes by matching seed ids when available.

    Priority:
    - If there are common numeric seeds, pair those (one file per seed per side; if
      multiple files per seed, take them all cross-product).
    - Else, if both sides have only unseeded files (key None), pair them one-to-one
      in sorted order by filename up to min length.
    - Else, if one side has seeded and the other does not, pair unseeded files with
      all seeded files (best-effort pairing).
    """
    grouped_a = _collect_seeded_files(prefix_a)
    grouped_b = _collect_seeded_files(prefix_b)

    seeds_a = set(grouped_a.keys())
    seeds_b = set(grouped_b.keys())

    common_numeric_seeds = [s for s in seeds_a.intersection(seeds_b) if s is not None]

    pairs: List[Tuple[str, str]] = []

    if len(common_numeric_seeds) > 0:
        common_numeric_seeds.sort()
        for s in common_numeric_seeds:
            for fa in sorted(grouped_a.get(s, [])):
                for fb in sorted(grouped_b.get(s, [])):
                    pairs.append((fa, fb))
        return pairs

    # Fallbacks when no common numeric seeds
    files_a_none = sorted(grouped_a.get(None, []))
    files_b_none = sorted(grouped_b.get(None, []))

    if len(files_a_none) > 0 and len(files_b_none) > 0:
        # Pair in order
        for fa, fb in zip(files_a_none, files_b_none):
            pairs.append((fa, fb))
        return pairs

    # Mixed case: one side has seeds, the other does not
    if len(files_a_none) > 0 and len(grouped_b) > 0:
        seeded_b = [f for s, fs in grouped_b.items() if s is not None for f in fs]
        for fa in files_a_none:
            for fb in sorted(seeded_b):
                pairs.append((fa, fb))
        return pairs

    if len(files_b_none) > 0 and len(grouped_a) > 0:
        seeded_a = [f for s, fs in grouped_a.items() if s is not None for f in fs]
        for fb in files_b_none:
            for fa in sorted(seeded_a):
                pairs.append((fa, fb))
        return pairs

    return pairs


def compute_multiplier_estimate_by_loss(
    base_loss_prefix: str,
    second_loss_prefix: str,
    target_loss: float,
    loss_column: str = LOSS_COLUMN,
    verbose: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute the mean by-loss multiplier and its 95% CI across paired files.

    Returns (mean_multiplier, error_bar) or (None, None) if no pairs found.
    """
    pairs = _pair_seed_files(base_loss_prefix, second_loss_prefix)

    if verbose:
        print(
            f"\nPairing for prefixes '{base_loss_prefix}' vs '{second_loss_prefix}': {len(pairs)} pairs"
        )
        for a, b in pairs[:5]:
            print(f"  {os.path.basename(a)}  <->  {os.path.basename(b)}")
        if len(pairs) > 5:
            print("  ...")

    if len(pairs) == 0:
        print(
            f"No paired files found for prefixes '{base_loss_prefix}' and '{second_loss_prefix}'."
        )
        return None, None

    multipliers: List[float] = []
    for file_a, file_b in pairs:
        try:
            mult, _ = compute_multiplier_by_loss(
                file_a,
                file_b,
                target_loss=target_loss,
                loss_column=loss_column,
                verbose=False,
            )
            if np.isfinite(mult):
                multipliers.append(float(mult))
        except Exception as e:
            if verbose:
                print(f"  Skipping pair due to error: {e}")
            continue

    if len(multipliers) == 0:
        print(
            f"No valid multipliers computed for '{base_loss_prefix}' vs '{second_loss_prefix}'."
        )
        return None, None

    mean_multiplier = float(np.mean(multipliers))
    # 95% CI using standard error
    if len(multipliers) > 1:
        error_bar = float(
            np.std(multipliers, ddof=1) / np.sqrt(len(multipliers)) * 1.96
        )
    else:
        error_bar = 0.0

    return mean_multiplier, error_bar


# %%
# Setup plotting style and parameters
sns.set_style("ticks")

# ===== EASILY ADJUSTABLE PARAMETERS =====
confidence_intervals = CONFIDENCE_INTERVALS  # Set to False to hide error bars

# ===== EASILY ADJUSTABLE FONT SIZES =====
title_fontsize = TITLE_FONTSIZE
axis_label_fontsize = AXIS_LABEL_FONTSIZE
x_tick_label_fontsize = X_TICK_LABEL_FONTSIZE
y_tick_label_fontsize = Y_TICK_LABEL_FONTSIZE
legend_fontsize = LEGEND_FONTSIZE
value_label_fontsize = VALUE_LABEL_FONTSIZE
x_tick_rotation = X_TICK_ROTATION

# %%
# Compute by-loss multipliers (paired across seeds when available)
swiglu_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_swiglu", "alg_mult/64d_gelu", target_loss=TARGET_LOSS
)
print(swiglu_estimate)

# %%
swiglu_relu_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_swiglu", "alg_mult/64d_relu", target_loss=TARGET_LOSS
)
print(swiglu_relu_estimate)

# %%
gelu_relu_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_gelu", "alg_mult/64d_relu", target_loss=TARGET_LOSS
)
print(gelu_relu_estimate)

# %%
# rotary vs sinusoidal (compute and print, but keep manual override in plot if desired)
rotary_sinusoidal_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_rotary", "alg_mult/64d_sinusoidal", target_loss=TARGET_LOSS
)
print(rotary_sinusoidal_estimate)

learned_vs_sinusoidal_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_learned", "alg_mult/64d_sinusoidal", target_loss=TARGET_LOSS
)
print(learned_vs_sinusoidal_estimate)

# %%
# rotary vs learned
rotary_learned_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_rotary", "alg_mult/64d_learned", target_loss=TARGET_LOSS
)
print(rotary_learned_estimate)

# %%
# transformer vs lstm (manual placeholder; edit as needed)
# transformer_lstm_estimate = [1.0, 0]
transformer_lstm_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_swiglu",
    "lstm_scaling_study/64_correction_bs64",
    target_loss=TARGET_LOSS,
)
print(transformer_lstm_estimate)

# %%
# adam vs sgd estimate at bs64
adam_sgd_estimate = compute_multiplier_estimate_by_loss(
    "transformer_scaling/swiglu_64d_transformer_bs64",
    "sgd_scaling/64d_sgdbs64",
    target_loss=TARGET_LOSS,
)
print(adam_sgd_estimate)

# %%
# estimate of cosine warmup vs inverse_sqrt
cosine_inverse_sqrt_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_gelu", "alg_mult/64d_lr_inverse_sqrt", target_loss=TARGET_LOSS
)
print(cosine_inverse_sqrt_estimate)

cosine_v_linear_estimate = compute_multiplier_estimate_by_loss(
    "alg_mult/64d_gelu", "alg_mult/64d_linear_warmup", target_loss=TARGET_LOSS
)
print(cosine_v_linear_estimate)


# %%
# Create bar plot of compute multiplier estimates with error bars
estimates_data = [
    # ("Rotary vs Learned", rotary_learned_estimate),
    ("Rotary vs Sinusoidal", [1.4, 0]),  # manual override retained
    ("SwiGLU vs ReLU", [1.1, 0]),  # manual override retained
    ("GELU vs ReLU", gelu_relu_estimate),
    ("Transformer vs LSTM", transformer_lstm_estimate),
    ("Learned vs Sinusoidal", learned_vs_sinusoidal_estimate),
    ("Cosine Warmup vs Inverse Sqrt", cosine_inverse_sqrt_estimate),
    ("Cosine Warmup vs Linear", cosine_v_linear_estimate),
    ("Adam vs SGD (bs=64)", adam_sgd_estimate),
]

# Filter out None estimates and separate labels, multipliers, and error bars
valid_estimates = [
    (label, data) for label, data in estimates_data if data[0] is not None
]

if valid_estimates:
    # Sort by multiplier so that the highest bar is on the right
    valid_estimates_sorted = sorted(valid_estimates, key=lambda x: x[1][0])

    labels = [item[0] for item in valid_estimates_sorted]
    multipliers = [item[1][0] for item in valid_estimates_sorted]
    error_bars = [item[1][1] for item in valid_estimates_sorted]

    # Choose color palette length to match number of bars (repeat if more than 3)
    palette = sns.color_palette("viridis", n_colors=len(labels))
    bar_colors = palette[: len(labels)]

    plt.figure(figsize=(10, 6))

    # Conditionally add error bars based on confidence_intervals parameter
    if confidence_intervals:
        bars = plt.bar(
            labels,
            multipliers,
            yerr=error_bars,
            capsize=5,
            color=bar_colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=1,
        )
    else:
        bars = plt.bar(
            labels,
            multipliers,
            color=bar_colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=1,
        )

    plt.ylabel("(CEG) Multiplier Estimate", fontsize=axis_label_fontsize)
    plt.xlabel("Improvement Type", fontsize=axis_label_fontsize)
    plt.title(
        "Compute Multiplier Estimates for Various Improvements",
        fontsize=title_fontsize,
        fontweight="bold",
    )
    # Increase the font size of the x-tick labels for visibility
    plt.xticks(
        rotation=x_tick_rotation,
        ha="right",
        fontsize=x_tick_label_fontsize,  # Use separate parameter
        fontweight="bold",
    )
    plt.yticks(fontsize=y_tick_label_fontsize)  # Use separate parameter
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    for i, (bar, multiplier, error) in enumerate(zip(bars, multipliers, error_bars)):
        if confidence_intervals:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + error + 0.05,
                f"{multiplier:.2f}±{error:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_label_fontsize,
                fontweight="bold",
                color="black",
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f"{multiplier:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_label_fontsize,
                fontweight="bold",
                color="black",
            )

    # Add a horizontal line at y=1 for reference (no improvement)
    plt.axhline(
        y=1, color="red", linestyle="--", alpha=0.5, label="No improvement (1.0x)"
    )
    plt.yscale("log")
    plt.ylim(0, 4)
    plt.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary of Compute Multiplier (by-loss) Estimates:")
    print("=" * 50)
    for label, multiplier, error in zip(labels, multipliers, error_bars):
        improvement = (
            ((multiplier - 1) * 100)
            if multiplier > 1
            else (-(1 / multiplier - 1) * 100)
        )
        if confidence_intervals:
            print(
                f"{label}: {multiplier:.3f}x"
                + (f" ± {error:.3f}" if error is not None else "")
                + f" ({improvement:+.1f}% compute efficiency)"
            )
        else:
            print(
                f"{label}: {multiplier:.3f}x ({improvement:+.1f}% compute efficiency)"
            )
else:
    print("No valid estimates found to plot.")


# %%
