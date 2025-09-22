# %%
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# %%

IRREDUCIBLE_LOSS = 1.8
GAMMA = 0.155


def final_loss(file_name):
    file = pd.read_csv(file_name)
    return file["validation_loss"].iloc[-1]


def compute_effect(loss_1, loss_2):
    irreducible = 1.7
    return np.exp(
        -(np.log(loss_1 - irreducible) - np.log(loss_2 - irreducible)) / 0.155
    )


def loss_statistics(file_prefix):
    # Search for files matching the prefix in the specified directory
    # This will match both 'prefix.csv' and 'prefix_seed.csv'
    import os

    # Check if we're in the experimental_analysis directory or the main project directory
    if os.path.exists("../experimental_data_folder/"):
        base_path = "../experimental_data_folder/"
    else:
        base_path = "experimental_data_folder/"

    file_pattern = f"{base_path}{file_prefix}*.csv"
    print(f"Searching for files with pattern: {file_pattern}")
    all_matching_files = glob.glob(file_pattern)

    # Filter to only include files with seed numbers at the end (e.g., _123.csv, _456.csv)
    import re

    seed_pattern = re.compile(r"_\d+\.csv$")
    matching_files = [f for f in all_matching_files if seed_pattern.search(f)]

    print(
        f"Found {len(all_matching_files)} total files, {len(matching_files)} with seed numbers"
    )

    if not matching_files:
        print(f"No files with seed numbers found for prefix: {file_prefix}")
        return None, None

    losses = []
    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path)
            if "Validation Loss" in df.columns:
                loss_val = df["Validation Loss"].iloc[-1]
            elif "validation_loss" in df.columns:
                loss_val = df["validation_loss"].iloc[-1]
            else:
                print(
                    f"Warning: 'Validation Loss' or 'validation_loss' column not found in {file_path}"
                )
                continue

            # Only add non-nan values
            if not np.isnan(loss_val):
                losses.append(loss_val)
            else:
                print(f"Warning: Found nan validation loss in {file_path}, skipping")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not losses:
        print(f"No valid loss data found for prefix: {file_prefix}")
        return None, None

    average_loss = np.mean(losses)
    confidence_interval = np.std(losses) / np.sqrt(len(losses)) * 1.96

    return average_loss, confidence_interval


# find compute multiplier
def compute_multiplier(loss_1, loss_2, irreducible=IRREDUCIBLE_LOSS, C=GAMMA):
    return np.exp(-(np.log(loss_1 - irreducible) - np.log(loss_2 - irreducible)) / C)


def compute_multiplier_estimate(
    base_loss_prefix, second_loss_prefix, irreducible=IRREDUCIBLE_LOSS, C=GAMMA
):
    """
    Computes the multiplier estimate and its adjusted error bar for two sets of experiments.

    Args:
        base_loss_prefix (str): The file prefix for the baseline loss data.
        second_loss_prefix (str): The file prefix for the second loss data.

    Returns:
        tuple: A tuple containing (multiplier_estimate, adjusted_error_bar).
               Returns (None, None) if loss data cannot be retrieved.
    """
    # Get average loss and confidence interval for base loss
    avg_loss_1, ci_1 = loss_statistics(base_loss_prefix)
    if avg_loss_1 is None:
        print(f"Could not retrieve statistics for base loss prefix: {base_loss_prefix}")
        return None, None

    # Get average loss and confidence interval for second loss
    avg_loss_2, ci_2 = loss_statistics(second_loss_prefix)
    if avg_loss_2 is None:
        print(
            f"Could not retrieve statistics for second loss prefix: {second_loss_prefix}"
        )
        return None, None

    # Compute the multiplier estimate
    multiplier_estimate = compute_multiplier(avg_loss_1, avg_loss_2, irreducible, C)

    # Calculate the adjusted error bar using error propagation

    # Derivatives for f = -(log(L1 - irr) - log(L2 - irr)) / C
    dL1_term_sq = (ci_1 / (C * (avg_loss_1 - irreducible))) ** 2
    dL2_term_sq = (ci_2 / (C * (avg_loss_2 - irreducible))) ** 2

    # Error in f (df)
    df = np.sqrt(dL1_term_sq + dL2_term_sq)

    # Error in M = exp(f) is M * df
    adjusted_error_bar = multiplier_estimate * df

    return [multiplier_estimate, adjusted_error_bar]


# %%
# exampple usage
loss_statistics("alg_mult/64d_swiglu")

# %%

# if more compute is used in a given algorithm


swiglu_estimate = compute_multiplier_estimate(
    "alg_mult/64d_swiglu", "alg_mult/64d_gelu"
)
print(swiglu_estimate)
# %%

swiglu_relu_estimate = compute_multiplier_estimate(
    "alg_mult/64d_swiglu", "alg_mult/64d_relu"
)


print(swiglu_relu_estimate)
# %%
gelu_relu_estimate = compute_multiplier_estimate(
    "alg_mult/64d_gelu", "alg_mult/64d_relu"
)
print(gelu_relu_estimate)
# rotary vs sinusoidal
rotary_sinusoidal_estimate = compute_multiplier_estimate(
    "alg_mult/64d_rotary", "alg_mult/64d_sinusoidal"
)
print(rotary_sinusoidal_estimate)
# %%
learned_vs_sinusoidal_estimate = compute_multiplier_estimate(
    "alg_mult/64d_learned", "alg_mult/64d_sinusoidal"
)
print(learned_vs_sinusoidal_estimate)
# %%

# rotay vs learend
rotary_learned_estimate = compute_multiplier_estimate(
    "alg_mult/64d_rotary", "alg_mult/64d_learned"
)
print(rotary_learned_estimate)

# transformer vs lstm
transformer_lstm_estimate = [compute_multiplier(5.2039, 5.8135), 0]
# learned vs sinusoidal
print(transformer_lstm_estimate)

#%%
#historical anlaysis 
transformer_2022_2017_64 = [compute_multiplier(5.1735, 5.5102), 0]
print(transformer_2022_2017_64, "historical analysis")


transformer_2022_2017_64 = [compute_multiplier(4.5318, 5.1177), 0]
print(transformer_2022_2017_64, "historical analysis 128")


# %%
# estimate of csoine warmup vs inverss_sqrt
cosine_inverse_sqrt_estimate = compute_multiplier_estimate(
    "alg_mult/64d_gelu", "alg_mult/64d_lr_inverse_sqrt"
)
print(cosine_inverse_sqrt_estimate)
# %%


cosine_v_linear_estimate = compute_multiplier_estimate(
    "alg_mult/64d_gelu", "alg_mult/64d_linear_warmup"
)
print(cosine_v_linear_estimate, "cosine vs linear")


# %%
# adam_estimate = compute_multiplier_estimate(
#     "optimizer_experiments/32d_adam", "optimizer_experiments/32d_sgd"
# )
# rotary_estimate = compute_multiplier_estimate(
#     "pos_encoding/32d_rotary", "pos_encoding/32d_learned"
# )
# learned_estimate = compute_multiplier_estimate(
#     "pos_encoding/32d_learned", "pos_encoding/32d_sinusoidal"
# )
# # trans_lstm = compute_multiplier_estimate("Optimizer_Experiments/32d_adam", "lstm_optimizer/LSTMADAM")
# trans_lstm = compute_multiplier_estimate(
#     "Optimizer_Experiments/32d_adam", "LSTM_Hidden_Dim_Scaling/LSTM_16d"
# )
# if trans_lstm[0] is not None:
#     trans_lstm = [(10 / 3) * trans_lstm[0], (10 / 3) * trans_lstm[1]]
# print("SwiGLU estimate:", swiglu_estimate)
# print("Adam estimate:", adam_estimate)
# print("Rotary estimate:", rotary_estimate)
# print("Learned estimate:", learned_estimate)
# print("Transformer Estimate:", trans_lstm)

# %%
# Create bar plot of compute multiplier estimates with error bars

# Collect all estimates and their labels
estimates_data = [
    ("Rotary vs Learned", rotary_learned_estimate),
    ("Rotary vs Sinusoidal", rotary_sinusoidal_estimate),
    ("SwiGLU vs ReLU", swiglu_relu_estimate),
    ("GELU vs ReLU", gelu_relu_estimate),
    ("Transformer vs LSTM", transformer_lstm_estimate),
    ("Learned vs Sinusoidal", learned_vs_sinusoidal_estimate),
    ("Cosine Warmup vs Inverse Sqrt", cosine_inverse_sqrt_estimate),
    ("Cosine Warmup vs Linear", cosine_v_linear_estimate),
]

import seaborn as sns

# Set seaborn style and context
sns.set_style("ticks")
# sns.set_context("paper")

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

    plt.ylabel("(CEG) Multiplier Estimate", fontsize=12)
    plt.xlabel("Improvement Type", fontsize=12)
    plt.title(
        "Compute Multiplier Estimates for Various Improvements",
        fontsize=14,
        fontweight="bold",
    )
    # Increase the font size of the x-tick labels for visibility
    plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
    plt.yticks(fontsize=13)
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on top of bars
    for i, (bar, multiplier, error) in enumerate(zip(bars, multipliers, error_bars)):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + error + 0.05,
            f"{multiplier:.2f}±{error:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    # Add a horizontal line at y=1 for reference (no improvement)
    plt.axhline(
        y=1, color="red", linestyle="--", alpha=0.5, label="No improvement (1.0x)"
    )
    plt.yscale("log")
    plt.ylim(0, 4)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary of Compute Multiplier Estimates:")
    print("=" * 50)
    for label, multiplier, error in zip(labels, multipliers, error_bars):
        improvement = (
            ((multiplier - 1) * 100)
            if multiplier > 1
            else (-(1 / multiplier - 1) * 100)
        )
        print(
            f"{label}: {multiplier:.3f}x ± {error:.3f} ({improvement:+.1f}% compute efficiency)"
        )

else:
    print("No valid estimates found to plot.")

# %%
