# %%
import pandas as pd
import numpy as np
import glob

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

    return multiplier_estimate, adjusted_error_bar


# %%
# exampple usage
loss_statistics("Activation_Functions_Comparison/ReLU")

# %%

#if more compute is used in a given algorithm 



swiglu_estimate = compute_multiplier_estimate(
    "activation_function/swiglu", "activation_function/gelu"
)
adam_estimate = compute_multiplier_estimate(
    "optimizer_experiments/32d_adam", "optimizer_experiments/32d_sgd"
)
rotary_estimate = compute_multiplier_estimate(
    "pos_encoding/32d_rotary", "pos_encoding/32d_learned"
)
learned_estimate = compute_multiplier_estimate(
    "pos_encoding/32d_learned", "pos_encoding/32d_standard"
)




# %%
