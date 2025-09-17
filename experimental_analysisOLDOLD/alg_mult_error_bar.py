# %%
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# %%


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
    base_path = "../experimental_data_folder/"
    file_pattern = f"{base_path}{file_prefix}*.csv"
    print(f"Searching for files with pattern: {file_pattern}")
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No files found for prefix: {file_prefix}")
        return None, None

    losses = []
    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path)
            if "Validation Loss" in df.columns:
                losses.append(df["Validation Loss"].iloc[-1])
            elif "validation_loss" in df.columns:
                losses.append(df["validation_loss"].iloc[-1])
            else:
                print(
                    f"Warning: 'Validation Loss' or 'validation_loss' column not found in {file_path}"
                )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not losses:
        print(f"No valid loss data found for prefix: {file_prefix}")
        return None, None

    average_loss = np.mean(losses)
    confidence_interval = np.std(losses) / np.sqrt(len(losses)) * 1.96

    return average_loss, confidence_interval


# find compute multiplier
def compute_multiplier(loss_1, loss_2, irreducible=1.7, C=0.155):
    return np.exp(-(np.log(loss_1 - irreducible) - np.log(loss_2 - irreducible)) / C)


def compute_multiplier_estimate(
    base_loss_prefix, second_loss_prefix, irreducible=1.7, C=0.155
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
loss_statistics("Activation_Functions_Comparison/ReLU")

# %%

# if more compute is used in a given algorithm


swiglu_estimate = compute_multiplier_estimate(
    "activation_function/SwiGLU", "activation_function/GELU"
)
adam_estimate = compute_multiplier_estimate(
    "optimizer_experiments/32d_adam", "optimizer_experiments/32d_sgd"
)
rotary_estimate = compute_multiplier_estimate(
    "pos_encoding/32d_rotary", "pos_encoding/32d_learned"
)
learned_estimate = compute_multiplier_estimate(
    "pos_encoding/32d_learned", "pos_encoding/32d_sinusoidal"
)
# trans_lstm = compute_multiplier_estimate("Optimizer_Experiments/32d_adam", "lstm_optimizer/LSTMADAM")
trans_lstm = compute_multiplier_estimate("Optimizer_Experiments/32d_adam", "LSTM_Hidden_Dim_Scaling/LSTM_16d")
if trans_lstm[0] is not None:
    trans_lstm = [(10/3) * trans_lstm[0], (10/3) * trans_lstm[1]]
print("SwiGLU estimate:", swiglu_estimate)
print("Adam estimate:", adam_estimate)
print("Rotary estimate:", rotary_estimate)
print("Learned estimate:", learned_estimate)
print("Transformer Estimate:", trans_lstm)

# %%
# Create bar plot of compute multiplier estimates with error bars

# Collect all estimates and their labels
estimates_data = [
    ("SwiGLU vs GELU", swiglu_estimate),
    ("Adam vs SGD", adam_estimate),
    ("Rotary vs Learned", rotary_estimate),
    ("Learned vs Standard", learned_estimate),
    ("Transformer vs LSTM", trans_lstm)
]

# Filter out None estimates and separate labels, multipliers, and error bars
valid_estimates = [
    (label, data) for label, data in estimates_data if data[0] is not None
]

if valid_estimates:
    # Sort by multiplier so that the highest bar is on the right
    # Each item: (label, (multiplier, error_bar))
    valid_estimates_sorted = sorted(valid_estimates, key=lambda x: x[1][0])

    labels = [item[0] for item in valid_estimates_sorted]
    multipliers = [item[1][0] for item in valid_estimates_sorted]
    error_bars = [item[1][1] for item in valid_estimates_sorted]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        labels,
        multipliers,
        yerr=error_bars,
        capsize=5,
        color=["skyblue", "lightcoral", "lightgreen", "gold"][: len(labels)],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Customize the plot
    plt.ylabel("Compute Multiplier Estimate", fontsize=12)
    plt.xlabel("Improvement Type", fontsize=12)
    plt.title(
        "Compute Multiplier Estimates for Various Improvements",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
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
        )

    # Add a horizontal line at y=1 for reference (no improvement)
    plt.axhline(
        y=1, color="red", linestyle="--", alpha=0.5, label="No improvement (1.0x)"
    )
    plt.legend()

    # Adjust layout and display
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
