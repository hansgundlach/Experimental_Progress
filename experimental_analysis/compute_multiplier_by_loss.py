# %%

#!/usr/bin/env python3
"""
Compute multiplier analysis by comparing compute required to reach same loss values.

This script computes how much less compute model A needs compared to model B
to reach the same validation loss value.
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import seaborn as sns
import os

# %%


def find_closest_loss_row(
    df: pd.DataFrame, target_loss: float, loss_column: str = "validation_loss"
) -> Tuple[int, float, float]:
    """
    Find the row with validation loss closest to the target loss.

    Args:
        df: DataFrame with training data
        target_loss: Target loss value to find
        loss_column: Name of the loss column to search in

    Returns:
        Tuple of (row_index, actual_loss_found, compute_at_that_point)
    """
    if loss_column not in df.columns:
        raise ValueError(
            f"Column '{loss_column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    if "total_flops_profiler" not in df.columns:
        raise ValueError(
            f"Column 'total_flops_profiler' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # Remove rows with NaN losses
    valid_df = df.dropna(subset=[loss_column])

    if len(valid_df) == 0:
        raise ValueError(f"No valid (non-NaN) values found in {loss_column} column")

    # Find the row with loss closest to target
    loss_differences = np.abs(valid_df[loss_column] - target_loss)
    closest_idx = loss_differences.idxmin()

    actual_loss = valid_df.loc[closest_idx, loss_column]
    compute_value = valid_df.loc[closest_idx, "total_flops_profiler"]

    return closest_idx, actual_loss, compute_value


def compute_multiplier_closest_approach(
    input_a: Union[str, Tuple[float, float, float]],
    input_b: Union[str, Tuple[float, float, float], None] = None,
    E: Optional[float] = None,
    A: Optional[float] = None,
    alpha: Optional[float] = None,
    compute_column: str = "total_flops_profiler",
    loss_column: str = "validation_loss",
    verbose: bool = True,
    compute_range: Optional[Tuple[float, float]] = None,
    num_samples: int = 1000,
) -> Tuple[float, dict]:
    """
    Find the closest approach point between two curves (empirical or power law),
    then compute the compute multiplier at that point.

    The power law fit has the form: Loss = E + A * C^(-alpha)

    Usage modes:
    1. CSV vs Power Law (legacy mode):
       compute_multiplier_closest_approach(csv_file, E=E, A=A, alpha=alpha)

    2. CSV vs Power Law (new mode):
       compute_multiplier_closest_approach(csv_file, (E, A, alpha))

    3. Power Law vs Power Law:
       compute_multiplier_closest_approach((E1, A1, alpha1), (E2, A2, alpha2))

    For CSV vs Power Law:
    - For each point on the empirical curve (C_empirical, L_empirical):
      - We solve for C_powerlaw where: E + A * C_powerlaw^(-alpha) = L_empirical
      - This gives: C_powerlaw = (A / (L_empirical - E))^(1/alpha)
      - We compute the x-axis distance: |C_empirical - C_powerlaw|
    - The closest approach is the point with minimum x-axis distance.
    - The multiplier is: C_powerlaw / C_empirical at closest approach point.

    For Power Law vs Power Law:
    - We sample compute values across a range
    - For each C1 on power law 1: L1 = E1 + A1 * C1^(-alpha1)
      - We solve for C2 on power law 2: E2 + A2 * C2^(-alpha2) = L1
      - We compute the x-axis distance: |C1 - C2|
    - The closest approach is the point with minimum x-axis distance.
    - The multiplier is: C2 / C1 at closest approach point.

    Args:
        input_a: Either a CSV file path (str) or power law parameters (E, A, alpha) tuple
        input_b: Either power law parameters (E, A, alpha) tuple or None
                 If None, use E, A, alpha parameters instead
        E: Irreducible loss parameter (legacy parameter, used if input_b is None)
        A: Amplitude parameter (legacy parameter, used if input_b is None)
        alpha: Exponent parameter (legacy parameter, used if input_b is None)
        compute_column: Name of compute column (default: 'total_flops_profiler')
        loss_column: Name of loss column (default: 'validation_loss')
        verbose: Whether to print detailed information
        compute_range: Tuple (min_compute, max_compute) for power law vs power law comparison
                       If None, automatically determined from the data/power laws
        num_samples: Number of samples for power law vs power law comparison

    Returns:
        Tuple of (multiplier, details_dict) where:
        - multiplier: compute_b / compute_a at closest approach
        - details_dict: Dictionary with detailed information about the comparison
    """

    try:
        # Determine input types and normalize parameters
        is_csv_a = isinstance(input_a, str)
        is_powerlaw_a = isinstance(input_a, tuple) and len(input_a) == 3

        # Handle input_b - could be tuple, or None (legacy mode with E, A, alpha)
        if input_b is None:
            # Legacy mode: CSV vs power law using E, A, alpha parameters
            if E is None or A is None or alpha is None:
                raise ValueError(
                    "Must provide either input_b tuple or E, A, alpha parameters"
                )
            is_powerlaw_b = True
            powerlaw_b = (E, A, alpha)
        else:
            is_powerlaw_b = isinstance(input_b, tuple) and len(input_b) == 3
            powerlaw_b = input_b

        if not is_csv_a and not is_powerlaw_a:
            raise ValueError(
                "input_a must be either a CSV file path (str) or power law parameters (E, A, alpha) tuple"
            )

        if not is_powerlaw_b:
            raise ValueError("input_b must be power law parameters (E, A, alpha) tuple")

        # Extract power law B parameters
        E_b, A_b, alpha_b = powerlaw_b

        # Case 1: CSV vs Power Law
        if is_csv_a:
            csv_file = input_a

            # Read the CSV file
            df = pd.read_csv(csv_file)

            if verbose:
                print(f"Loaded {csv_file}: {len(df)} rows")

            # Validate columns exist
            if loss_column not in df.columns:
                raise ValueError(
                    f"Column '{loss_column}' not found in DataFrame. Available columns: {list(df.columns)}"
                )
            if compute_column not in df.columns:
                raise ValueError(
                    f"Column '{compute_column}' not found in DataFrame. Available columns: {list(df.columns)}"
                )

            # Remove rows with NaN values
            valid_df = df.dropna(subset=[loss_column, compute_column])

            if len(valid_df) == 0:
                raise ValueError(f"No valid (non-NaN) values found in required columns")

            # For each empirical point, find the corresponding power law compute
            distances = []
            powerlaw_computes = []

            for idx, row in valid_df.iterrows():
                C_a = float(row[compute_column])
                L_a = float(row[loss_column])

                # Check if loss is above irreducible loss
                if L_a <= E_b:
                    # Loss is at or below irreducible loss, power law cannot reach it
                    # Skip this point
                    distances.append(np.inf)
                    powerlaw_computes.append(np.nan)
                    continue

                # Solve for C_b: E_b + A_b * C_b^(-alpha_b) = L_a
                # => C_b = (A_b / (L_a - E_b))^(1/alpha_b)
                try:
                    C_b = (A_b / (L_a - E_b)) ** (1.0 / alpha_b)

                    # Compute x-axis distance
                    distance = abs(C_a - C_b)

                    distances.append(distance)
                    powerlaw_computes.append(C_b)
                except (ZeroDivisionError, ValueError, OverflowError):
                    distances.append(np.inf)
                    powerlaw_computes.append(np.nan)

            # Add columns to dataframe
            valid_df = valid_df.copy()
            valid_df["powerlaw_b_compute"] = powerlaw_computes
            valid_df["x_distance"] = distances

            # Find the point with minimum distance
            finite_distances = valid_df[np.isfinite(valid_df["x_distance"])]

            if len(finite_distances) == 0:
                raise ValueError(
                    "No valid closest approach points found (all distances are infinite)"
                )

            closest_idx = finite_distances["x_distance"].idxmin()

            # Get values at closest approach
            closest_row = valid_df.loc[closest_idx]
            C_a_closest = float(closest_row[compute_column])
            L_closest = float(closest_row[loss_column])
            C_b_closest = float(closest_row["powerlaw_b_compute"])
            distance_closest = float(closest_row["x_distance"])

            # Compute multiplier (power law B / CSV A)
            multiplier = C_b_closest / C_a_closest

            # Prepare detailed results
            details = {
                "comparison_type": "csv_vs_powerlaw",
                "input_a": {
                    "type": "csv",
                    "file": csv_file,
                },
                "input_b": {
                    "type": "powerlaw",
                    "E": E_b,
                    "A": A_b,
                    "alpha": alpha_b,
                },
                "closest_approach": {
                    "loss": L_closest,
                    "compute_a": C_a_closest,
                    "compute_b": C_b_closest,
                    "x_distance": distance_closest,
                    "row_index": closest_idx,
                },
                "multiplier": multiplier,
                "compute_ratio_b_to_a": multiplier,
            }

            if verbose:
                print(f"\nComparison: CSV vs Power Law")
                print(f"CSV: {csv_file}")
                print(f"Power Law B: Loss = {E_b:.4f} + {A_b:.4e} * C^(-{alpha_b:.4f})")
                print("=" * 60)
                print(f"\nClosest Approach Point:")
                print(f"  Validation Loss: {L_closest:.4f}")
                print(f"  CSV Compute (A): {C_a_closest:.2e} FLOPs")
                print(f"  Power Law Compute (B): {C_b_closest:.2e} FLOPs")
                print(f"  X-axis Distance: {distance_closest:.2e} FLOPs")
                print(f"  Row index: {closest_idx}")
                print(f"\nCompute Multiplier: {multiplier:.3f}x")

                if multiplier > 1:
                    print(
                        f"Power law B requires {multiplier:.3f}x MORE compute than CSV A at this loss"
                    )
                else:
                    print(
                        f"Power law B requires {1/multiplier:.3f}x LESS compute than CSV A at this loss"
                    )

            return multiplier, details

        # Case 2: Power Law vs Power Law
        else:
            E_a, A_a, alpha_a = input_a

            # Determine compute range for sampling
            if compute_range is None:
                # Auto-determine range based on power law parameters
                # Sample from where loss is significant (e.g., 2x irreducible loss) to very high compute
                # For power law A: at 2x irreducible loss: E_a + A_a * C^(-alpha_a) = 2 * E_a
                # => C = (A_a / E_a)^(1/alpha_a)
                try:
                    C_min = (A_a / max(E_a, 0.1)) ** (
                        1.0 / alpha_a
                    ) / 100  # Start earlier
                    C_max = (A_a / max(E_a, 0.1)) ** (
                        1.0 / alpha_a
                    ) * 100  # Go much further
                except (ZeroDivisionError, ValueError, OverflowError):
                    # Fallback to reasonable defaults
                    C_min = 1e14
                    C_max = 1e19
                compute_range = (C_min, C_max)

            C_min, C_max = compute_range

            if verbose:
                print(f"Sampling compute range: {C_min:.2e} to {C_max:.2e} FLOPs")

            # Sample compute values logarithmically
            C_samples_a = np.logspace(np.log10(C_min), np.log10(C_max), num_samples)

            distances = []
            C_b_values = []
            L_values = []

            for C_a in C_samples_a:
                # Calculate loss at C_a on power law A
                try:
                    L_a = E_a + A_a * (C_a ** (-alpha_a))

                    # Check if power law B can reach this loss
                    if L_a <= E_b:
                        distances.append(np.inf)
                        C_b_values.append(np.nan)
                        L_values.append(L_a)
                        continue

                    # Solve for C_b: E_b + A_b * C_b^(-alpha_b) = L_a
                    # => C_b = (A_b / (L_a - E_b))^(1/alpha_b)
                    C_b = (A_b / (L_a - E_b)) ** (1.0 / alpha_b)

                    # Compute x-axis distance
                    distance = abs(C_a - C_b)

                    distances.append(distance)
                    C_b_values.append(C_b)
                    L_values.append(L_a)

                except (ZeroDivisionError, ValueError, OverflowError):
                    distances.append(np.inf)
                    C_b_values.append(np.nan)
                    L_values.append(np.nan)

            # Find minimum distance
            finite_indices = np.isfinite(distances)
            if not np.any(finite_indices):
                raise ValueError(
                    "No valid closest approach points found (all distances are infinite)"
                )

            distances_array = np.array(distances)
            min_idx = np.nanargmin(np.where(finite_indices, distances_array, np.inf))

            # Get values at closest approach
            C_a_closest = float(C_samples_a[min_idx])
            C_b_closest = float(C_b_values[min_idx])
            L_closest = float(L_values[min_idx])
            distance_closest = float(distances[min_idx])

            # Compute multiplier (power law B / power law A)
            multiplier = C_b_closest / C_a_closest

            # Prepare detailed results
            details = {
                "comparison_type": "powerlaw_vs_powerlaw",
                "input_a": {
                    "type": "powerlaw",
                    "E": E_a,
                    "A": A_a,
                    "alpha": alpha_a,
                },
                "input_b": {
                    "type": "powerlaw",
                    "E": E_b,
                    "A": A_b,
                    "alpha": alpha_b,
                },
                "closest_approach": {
                    "loss": L_closest,
                    "compute_a": C_a_closest,
                    "compute_b": C_b_closest,
                    "x_distance": distance_closest,
                },
                "multiplier": multiplier,
                "compute_ratio_b_to_a": multiplier,
                "sampling_info": {
                    "compute_range": compute_range,
                    "num_samples": num_samples,
                },
            }

            if verbose:
                print(f"\nComparison: Power Law A vs Power Law B")
                print(f"Power Law A: Loss = {E_a:.4f} + {A_a:.4e} * C^(-{alpha_a:.4f})")
                print(f"Power Law B: Loss = {E_b:.4f} + {A_b:.4e} * C^(-{alpha_b:.4f})")
                print("=" * 60)
                print(f"\nClosest Approach Point:")
                print(f"  Validation Loss: {L_closest:.4f}")
                print(f"  Power Law A Compute: {C_a_closest:.2e} FLOPs")
                print(f"  Power Law B Compute: {C_b_closest:.2e} FLOPs")
                print(f"  X-axis Distance: {distance_closest:.2e} FLOPs")
                print(f"\nCompute Multiplier: {multiplier:.3f}x")

                if multiplier > 1:
                    print(
                        f"Power law B requires {multiplier:.3f}x MORE compute than Power law A at this loss"
                    )
                else:
                    print(
                        f"Power law B requires {1/multiplier:.3f}x LESS compute than Power law A at this loss"
                    )

            return multiplier, details

    except Exception as e:
        print(f"Error computing closest approach multiplier: {e}")
        raise


def _get_compute_from_input(
    input_value: Union[str, float, int],
    target_loss: float,
    loss_column: str = "validation_loss",
    verbose: bool = True,
) -> Tuple[float, dict]:
    """
    Helper function to get compute value from either a CSV file or a numeric value.

    Args:
        input_value: Either a CSV file path (str) or a numeric compute value (float/int)
        target_loss: Target loss value (only used if input_value is a CSV)
        loss_column: Name of the loss column (only used if input_value is a CSV)
        verbose: Whether to print detailed information

    Returns:
        Tuple of (compute_value, details_dict)
    """
    # Check if input is a number
    if isinstance(input_value, (int, float)):
        compute_value = float(input_value)
        details = {
            "is_numeric": True,
            "compute": compute_value,
            "source": f"Direct numeric value: {compute_value:.2e}",
        }
        if verbose:
            print(f"Using direct numeric compute value: {compute_value:.2e} FLOPs")
        return compute_value, details

    # Otherwise, treat as CSV file path
    if not isinstance(input_value, str):
        raise ValueError(
            f"Input must be either a CSV file path (str) or a numeric value (int/float), got {type(input_value)}"
        )

    # Check if file exists
    if not os.path.exists(input_value):
        raise FileNotFoundError(f"CSV file not found: {input_value}")

    # Read CSV and find closest loss point
    df = pd.read_csv(input_value)
    if verbose:
        print(f"Loaded {input_value}: {len(df)} rows")

    idx, actual_loss, compute_value = find_closest_loss_row(
        df, target_loss, loss_column
    )

    details = {
        "is_numeric": False,
        "file": input_value,
        "closest_row": idx,
        "actual_loss": actual_loss,
        "compute": compute_value,
        "loss_difference": abs(actual_loss - target_loss),
        "source": f"CSV file: {input_value}",
    }

    if verbose:
        print(
            f"  Closest loss: {actual_loss:.4f} (diff: {abs(actual_loss - target_loss):.4f})"
        )
        print(f"  Compute: {compute_value:.2e} FLOPs")
        print(f"  Row index: {idx}")

    return compute_value, details


def compute_multiplier_by_loss(
    input_a: Union[str, float, int],
    input_b: Union[str, float, int],
    target_loss: float,
    loss_column: str = "validation_loss",
    verbose: bool = True,
) -> Tuple[float, dict]:
    """
    Compute how much less compute model A needs vs model B to reach the same loss.

    Args:
        input_a: Path to first CSV file (str) OR numeric compute value (float/int) for model A
        input_b: Path to second CSV file (str) OR numeric compute value (float/int) for model B
        target_loss: Target loss value to compare at (only used if inputs are CSV files)
        loss_column: Name of the loss column (default: 'validation_loss', only used if inputs are CSV files)
        verbose: Whether to print detailed information

    Returns:
        Tuple of (multiplier, details_dict) where:
        - multiplier: compute_B / compute_A (how many times less compute A needs)
        - details_dict: Dictionary with detailed information about the comparison

    Examples:
        # Both CSV files
        multiplier, details = compute_multiplier_by_loss('model_a.csv', 'model_b.csv', 5.0)

        # One CSV, one numeric value
        multiplier, details = compute_multiplier_by_loss('model_a.csv', 1e17, 5.0)

        # Both numeric values
        multiplier, details = compute_multiplier_by_loss(6e16, 1e17, 5.0)
    """

    try:
        # Get compute values from both inputs
        compute_a, details_a = _get_compute_from_input(
            input_a, target_loss, loss_column, verbose
        )
        compute_b, details_b = _get_compute_from_input(
            input_b, target_loss, loss_column, verbose
        )

        # Calculate multiplier (how many times less compute A needs vs B)
        multiplier = compute_b / compute_a

        # Prepare detailed results
        details = {
            "target_loss": target_loss,
            "model_a": details_a,
            "model_b": details_b,
            "multiplier": multiplier,
            "compute_ratio_b_to_a": multiplier,
            "percent_compute_reduction": (
                ((multiplier - 1) / multiplier) * 100 if multiplier > 1 else 0
            ),
        }

        if verbose:
            print(f"\nResults for target loss: {target_loss}")
            print("=" * 50)
            print(f"Model A:")
            if details_a.get("is_numeric"):
                print(f"  Compute: {compute_a:.2e} FLOPs (direct numeric value)")
            else:
                print(f"  Source: {details_a.get('file', 'Unknown')}")
                print(
                    f"  Closest loss: {details_a.get('actual_loss', 'N/A'):.4f} (diff: {details_a.get('loss_difference', 0):.4f})"
                )
                print(f"  Compute: {compute_a:.2e} FLOPs")
                print(f"  Row index: {details_a.get('closest_row', 'N/A')}")

            print(f"\nModel B:")
            if details_b.get("is_numeric"):
                print(f"  Compute: {compute_b:.2e} FLOPs (direct numeric value)")
            else:
                print(f"  Source: {details_b.get('file', 'Unknown')}")
                print(
                    f"  Closest loss: {details_b.get('actual_loss', 'N/A'):.4f} (diff: {details_b.get('loss_difference', 0):.4f})"
                )
                print(f"  Compute: {compute_b:.2e} FLOPs")
                print(f"  Row index: {details_b.get('closest_row', 'N/A')}")

            print(f"\nCompute Multiplier: {multiplier:.3f}x")
            if multiplier > 1:
                print(
                    f"Model A needs {multiplier:.3f}x LESS compute than Model B to reach loss {target_loss}"
                )
                print(
                    f"Compute reduction: {((multiplier - 1) / multiplier) * 100:.1f}%"
                )
            else:
                print(
                    f"Model A needs {1/multiplier:.3f}x MORE compute than Model B to reach loss {target_loss}"
                )

        return multiplier, details

    except Exception as e:
        print(f"Error computing multiplier: {e}")
        raise


def compare_multiple_loss_points(
    csv_file_a: str,
    csv_file_b: str,
    loss_points: list,
    loss_column: str = "validation_loss",
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compare compute multipliers across multiple loss values.

    Args:
        csv_file_a: Path to first CSV file
        csv_file_b: Path to second CSV file
        loss_points: List of loss values to compare at
        loss_column: Name of the loss column
        plot: Whether to create a plot

    Returns:
        DataFrame with results for each loss point
    """

    results = []

    for target_loss in loss_points:
        try:
            multiplier, details = compute_multiplier_by_loss(
                csv_file_a, csv_file_b, target_loss, loss_column, verbose=False
            )

            results.append(
                {
                    "target_loss": target_loss,
                    "multiplier": multiplier,
                    "model_a_actual_loss": details["model_a"].get(
                        "actual_loss", target_loss
                    ),
                    "model_b_actual_loss": details["model_b"].get(
                        "actual_loss", target_loss
                    ),
                    "model_a_compute": details["model_a"]["compute"],
                    "model_b_compute": details["model_b"]["compute"],
                    "compute_reduction_percent": details["percent_compute_reduction"],
                }
            )

        except Exception as e:
            print(f"Error at loss {target_loss}: {e}")
            continue

    results_df = pd.DataFrame(results)

    if plot and len(results_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_df["target_loss"],
            results_df["multiplier"],
            "o-",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Target Loss Value")
        plt.ylabel("Compute Multiplier (Model B / Model A)")
        plt.title(f"Compute Efficiency Comparison\n{csv_file_a} vs {csv_file_b}")
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=1, color="red", linestyle="--", alpha=0.5, label="No difference (1.0x)"
        )

        # Add value labels on points
        for _, row in results_df.iterrows():
            plt.annotate(
                f'{row["multiplier"]:.2f}x',
                (row["target_loss"], row["multiplier"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.legend()
        plt.tight_layout()
        plt.show()

    return results_df


def analyze_loss_ranges(
    csv_file_a: str,
    csv_file_b: str,
    num_points: int = 10,
    loss_column: str = "validation_loss",
) -> pd.DataFrame:
    """
    Automatically analyze compute multipliers across the overlapping loss range.

    Args:
        csv_file_a: Path to first CSV file
        csv_file_b: Path to second CSV file
        num_points: Number of loss points to test
        loss_column: Name of the loss column

    Returns:
        DataFrame with results across the loss range
    """

    # Read files to determine overlapping loss range
    df_a = pd.read_csv(csv_file_a)
    df_b = pd.read_csv(csv_file_b)

    # Get valid loss ranges (excluding NaN values)
    valid_losses_a = df_a[loss_column].dropna()
    valid_losses_b = df_b[loss_column].dropna()

    # Find overlapping range
    min_loss_a, max_loss_a = valid_losses_a.min(), valid_losses_a.max()
    min_loss_b, max_loss_b = valid_losses_b.min(), valid_losses_b.max()

    overlap_min = max(min_loss_a, min_loss_b)
    overlap_max = min(max_loss_a, max_loss_b)

    print(f"Loss range overlap: {overlap_min:.3f} to {overlap_max:.3f}")

    if overlap_min >= overlap_max:
        raise ValueError("No overlapping loss range found between the two files")

    # Generate loss points in the overlapping range
    loss_points = np.linspace(overlap_min, overlap_max, num_points)

    return compare_multiple_loss_points(
        csv_file_a, csv_file_b, loss_points, loss_column
    )


# %%


# %%
# target_loss = 5.2
# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = "../experimental_data_folder/alg_mult/64d_swiglu_123.csv"
# example_file_b = "../experimental_data_folder/alg_mult/64d_gelu_123.csv"  # hypothetical

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )
# %%
# rotary vs sinusoidal


# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = "../experimental_data_folder/alg_mult/64d_rotary_456.csv"
# example_file_b = (
#     "../experimental_data_folder/alg_mult/64d_sinusoidal_456.csv"  # hypothetical
# )

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )
# %%
# target_loss = 5.5

# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = "../experimental_data_folder/alg_mult/64d_lr_cosine_warmup.csv"
# example_file_b = (
#     "../experimental_data_folder/alg_mult/64d_lr_inverse_sqrt.csv"  # hypothetical
# )

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )
# %%

# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = "../experimental_data_folder/alg_mult/64d_rotary_456.csv"
# example_file_b = (
#     "../experimental_data_folder/alg_mult/64d_sinusoidal_456.csv"  # hypothetical
# )

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )

# %%


# "csv_path": "../experimental_data_folder/debug_historical_experiments/radford_32transformer_2018_bs64.csv",

#           "name": "64d transformer scaling further",
#         "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_64d_transformer_bs64.csv",

# target_loss = 5.1
# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = (
#     "../experimental_data_folder/transformer_scaling/swiglu_64d_transformer_bs64.csv"
# )
# example_file_b = "../experimental_data_folder/debug_historical_experiments/radford_64transformer_2018_bs64.csv"  # hypothetical

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )

# target_loss = 5.1
# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = (
#     "../experimental_data_folder/transformer_scaling/swiglu_64d_transformer_bs64.csv"
# )
# example_file_b = "../experimental_data_folder/debug_historical_experiments/radford_128.csv"  # hypothetical

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )


# %%
# OVerall multiplier

# target_loss = 5.3

# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_a = "../experimental_data_folder/alg_mult/64d_gelu_123.csv"
# example_file_b = (
#     "../experimental_data_folder/alg_mult/64d_lr_inverse_sqrt_456.csv"  # hypothetical
# )

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )
# %%


# target_loss = 5.9

# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_b = (
#     "../experimental_data_folder/historical_experiments/p64transformer_2017_bs64.csv"
# )
# example_file_a = "../experimental_data_folder/historical_experiments/64transformer_2022_bs64.csv"  # hypothetical

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )


# %%
# trying a diffenret historical run

# target_loss = 5.1

# # Example usage and testing
# # Example usage - you would replace these with actual file paths
# example_file_b = "../experimental_data_folder/debug_historical_experiments/radford_64transformer_2018_bs64.csv"
# example_file_a = (
#     "../experimental_data_folder/modern_scaling_study/64_modern.csv"  # hypothetical
# )

# multiplier, details = compute_multiplier_by_loss(
#     example_file_a,
#     example_file_b,  # Using same file for demo
#     target_loss,
#     verbose=True,
# )


# %%

# Commented out test/example code below - uncomment to test
# print("Compute Multiplier by Loss Analysis")
# print("==================================")

# # Single loss point comparison

# try:
#     multiplier, details = compute_multiplier_by_loss(
#         example_file_a,
#         example_file_b,  # Using same file for demo
#         target_loss,
#         verbose=True,
#     )
#     print(multiplier)
# except FileNotFoundError:
#     print(f"Example files not found. Please provide actual CSV file paths.")
# except Exception as e:
#     print(f"Demo failed: {e}")
#     print("\nTo use this script, call the functions with your actual CSV file paths:")
#     print(
#         "multiplier, details = compute_multiplier_by_loss('model_a.csv', 'model_b.csv', 6.0)"
#     )
#     print(
#         "results_df = analyze_loss_ranges('model_a.csv', 'model_b.csv', num_points=15)"
#     )

# %%


# %%
# Updated Generic Stacked Bar Plot Function with Correct Multiplicative Effects


def create_stacked_comparison_plot(
    stacked_groups,
    comparison_data,
    title="Stacked vs Single Comparison",
    bar_order=None,
):
    """
    Create a bar plot with multiple stacked bars and multiple single-value bars.
    Each stacked bar shows cumulative multiplicative effects.

    Args:
        stacked_groups: List of lists, where each inner list contains tuples
                       (name, value, color) for one stacked bar.
                       Example: [[("A", 1.2, "#fff"), ("B", 1.3, "#eee")],
                                [("C", 1.1, "#ddd")]]
                       This creates 2 stacked bars.
        comparison_data: List of tuples (name, value, color) for non-stacked comparison bars
        title: Plot title
        bar_order: Optional list specifying the order of bars.
                   Format: ["stacked_0", "comparison_0", "stacked_1", "comparison_1", ...]
                   where "stacked_X" refers to stacked_groups[X] and "comparison_X" refers to comparison_data[X]
                   If None, uses default order: all stacked bars first, then all comparison bars
    """

    # ===== EASILY ADJUSTABLE FONT SIZES =====
    title_fontsize = 25
    axis_label_fontsize = 23
    tick_label_fontsize = 20
    ytick_label_fontsize = 20  # Added y-axis tick font size
    legend_fontsize = 20
    component_label_fontsize = 18
    value_label_fontsize = 20
    total_label_fontsize = 18
    # ========================================

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("viridis")

    # Set up bar positions and ordering
    n_stacked_groups = len(stacked_groups)
    n_comparison = len(comparison_data)
    total_bars = n_stacked_groups + n_comparison

    # Handle custom bar ordering
    if bar_order is None:
        # Default order: all stacked bars first, then all comparison bars
        bar_order = [f"stacked_{i}" for i in range(n_stacked_groups)] + [
            f"comparison_{i}" for i in range(n_comparison)
        ]

    # Validate bar_order
    valid_orders = [f"stacked_{i}" for i in range(n_stacked_groups)] + [
        f"comparison_{i}" for i in range(n_comparison)
    ]
    if not all(order in valid_orders for order in bar_order):
        raise ValueError(f"Invalid bar_order. Must contain only: {valid_orders}")
    if len(bar_order) != total_bars:
        raise ValueError(
            f"bar_order must have {total_bars} elements, got {len(bar_order)}"
        )

    x_positions = list(range(total_bars))

    # Create labels for each bar in the specified order
    labels = []
    for order in bar_order:
        if order.startswith("stacked_"):
            idx = int(order.split("_")[1])
            components_data = stacked_groups[idx]
            stacked_label = "Stacked:\n" + " \n × ".join(
                [comp[0] for comp in components_data]
            )
            labels.append(stacked_label)
        elif order.startswith("comparison_"):
            idx = int(order.split("_")[1])
            comparison_label = comparison_data[idx][0]
            labels.append(comparison_label)

    plt.figure(figsize=(16, 8))  # Wide figure for two-column layout

    # Plot bars in the specified order
    all_max_values = []  # Track all heights for y-axis scaling

    for bar_idx, order in enumerate(bar_order):
        if order.startswith("stacked_"):
            stacked_idx = int(order.split("_")[1])
            components_data = stacked_groups[stacked_idx]

            # Calculate cumulative multiplicative heights for this stacked bar
            cumulative_heights = [0.0]  # Start at 0
            running_product = 1.0  # Track the running product of multipliers

            for comp in components_data:
                multiplier = comp[1]
                running_product *= multiplier
                cumulative_heights.append(running_product)

            total_stacked_height = cumulative_heights[-1]
            all_max_values.append(total_stacked_height)

            # Generate viridis colors for components in this bar
            viridis_colors = sns.color_palette("viridis", n_colors=len(components_data))

            # Plot stacked components with cumulative multiplicative effects
            for i, (name, value, color) in enumerate(components_data):
                bottom = cumulative_heights[i]
                top = cumulative_heights[i + 1]
                segment_height = top - bottom

                # Use viridis colors instead of provided colors
                viridis_color = viridis_colors[i]
                bar = plt.bar(
                    x_positions[bar_idx],
                    segment_height,
                    bottom=bottom,
                    color=viridis_color,
                    width=0.6,
                    label=(
                        f"{name} ({value:.2f}x)" if bar_idx == 0 else ""
                    ),  # Only label first occurrence
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add component labels in the middle of each segment
                if segment_height > 0.1:  # Only add label if segment is large enough
                    # Position text in the middle of the segment
                    text_y = bottom + segment_height / 2

                    # Ensure text is visible on log scale (at least at y=1.1)
                    # But only adjust if the segment actually crosses the y=1 threshold
                    if bottom < 1.0 and text_y < 1.1 and top > 1.0:
                        text_y = 1.1

                    plt.text(
                        x_positions[bar_idx],
                        text_y,
                        f"{name}\n{value:.2f}x",
                        ha="center",
                        va="center",
                        fontsize=component_label_fontsize,
                        fontweight="bold",
                        color="white" if i % 2 == 0 else "black",
                    )

            # Add total height label for this stacked bar
            plt.text(
                x_positions[bar_idx],
                total_stacked_height + 0.2,
                f"Total: {total_stacked_height:.2f}x",
                ha="center",
                va="bottom",
                fontsize=total_label_fontsize,
                fontweight="bold",
                color="black",
            )

        elif order.startswith("comparison_"):
            comparison_idx = int(order.split("_")[1])
            name, value, color = comparison_data[comparison_idx]

            # Generate viridis colors for comparison bars
            comparison_viridis = sns.color_palette(
                "viridis", n_colors=len(comparison_data)
            )
            viridis_color = comparison_viridis[comparison_idx % len(comparison_viridis)]

            bar = plt.bar(
                x_positions[bar_idx],
                value,
                color=viridis_color,
                width=0.6,
                label=f"{name} ({value:.2f}x)",
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

            all_max_values.append(value)

            # Add value labels on top
            plt.text(
                x_positions[bar_idx],
                value + 0.1,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=value_label_fontsize,
                fontweight="bold",
            )

    # Customize plot with seaborn styling
    plt.xticks(
        x_positions, labels, rotation=0, ha="center", fontsize=tick_label_fontsize
    )
    plt.ylabel(
        "Compute Effect Multiplier", fontsize=axis_label_fontsize, fontweight="bold"
    )
    plt.title(title, fontsize=title_fontsize, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.yscale("log")

    # Set y-axis limits
    max_value = max(all_max_values) if all_max_values else 2.0
    plt.ylim(1, max_value * 1.4)

    # Fix y-axis tick font sizes - MUST be done AFTER yscale and ylim
    ax = plt.gca()

    # Control the number of y-axis ticks - set explicit tick locations
    # Adjust these values based on your data range:
    # Option 1: Minimal ticks (currently active)
    ax.set_yticks([1, 2])

    # Remove minor ticks to avoid extra tick marks
    ax.set_yticks([], minor=True)

    # Option 2: Show 1, 2, 5 pattern (uncomment to use instead)
    # ax.set_yticks([1, 2, 5])
    # ax.set_yticks([], minor=True)

    # Option 3: Include more values if needed (uncomment to use instead)
    # ax.set_yticks([1, 1.5, 2, 2.5, 3])
    # ax.set_yticks([], minor=True)

    ax.tick_params(axis="y", which="both", labelsize=ytick_label_fontsize)

    # Add legend below the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=legend_fontsize,
    )

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nSummary:")

    # Print each stacked group
    for group_idx, components_data in enumerate(stacked_groups):
        # Recalculate cumulative heights for printing
        cumulative_heights = [0.0]
        running_product = 1.0
        for comp in components_data:
            running_product *= comp[1]
            cumulative_heights.append(running_product)

        total_stacked_height = cumulative_heights[-1]

        print(f"\nStacked Group {group_idx + 1} - Total: {total_stacked_height:.2f}x")
        print("  Cumulative breakdown:")
        for i, (name, value, _) in enumerate(components_data):
            cumulative = cumulative_heights[i + 1]
            print(f"    After {name}: {cumulative:.2f}x (×{value:.2f})")

    if comparison_data:
        print(f"\nComparison values:")
        for name, value, _ in comparison_data:
            print(f"  {name}: {value:.2f}x")


# %%
# Example usage with placeholder data

# Example 1: 3 stacked components vs 2 comparison bars
# components_data = [
#     ("Adam", 2.5, "#3498db"),
#     ("Rotary", 1.8, "#85C1E9"),
#     ("SwiGLU", 1.4, "#2ECC71"),
# ]

# comparison_data = [
#     ("LSTM Baseline", 1.0, "#e74c3c"),
#     ("Simple Transformer", 3.2, "#f39c12"),
# ]

# create_stacked_comparison_plot(
#     components_data, comparison_data, "Advanced vs Simple Architectures"
# )


# components_data_2 = [
#     ("Rotary Encoding", 1.4, "#9b59b6"),
#     ("SwiGLU", 1.1, "#e67e22"),
#     ("Cosine Scheduler", 1.2, "#1abc9c"),
# ]
# %%
# Example 2: Multiple stacked bars and multiple comparison bars
# Now you can have multiple stacked groups!

# # Define stacked groups - each inner list is one stacked bar
# stacked_groups = [
#     # First stacked bar: Rotary + SwiGLU
#     [
#         ("Rotary Encoding", 1.4, "#9b59b6"),
#         ("SwiGLU", 1.1, "#e67e22"),
#     ],
#     # Second stacked bar: Another combination (optional, remove if not needed)
#     # [
#     #     ("Component A", 1.2, "#1abc9c"),
#     #     ("Component B", 1.15, "#3498db"),
#     # ],
# ]

# # Define comparison bars - these are single (non-stacked) bars
# comparison_data = [
#     ("Current vs 2017 Transformer", 1.677, "#e74c3c"),
#     # ("Ho et Al", 1.5, "#f39c12"),  # Uncomment to add more comparison bars
# ]

# # This creates: 1 stacked bar (with 2 components) + 1 non-stacked bar = 2 total bars
# # To get 4 bars, you could do:
# # - 2 stacked groups + 2 comparison bars, OR
# # - 1 stacked group + 3 comparison bars, OR
# # - 3 stacked groups + 1 comparison bar, etc.

# create_stacked_comparison_plot(
#     stacked_groups,
#     comparison_data,
#     "Transformer Components Effects vs Overall Increase",
# )

# %%
# Example 3: Create exactly 4 bars (2 stacked + 2 non-stacked)
# stacked_groups_4bars = [
#     # First stacked bar
#     [
#         ("Rotary Encoding", 1.4, "#9b59b6"),
#         ("SwiGLU", 1.1, "#e67e22"),
#     ],
#     # Second stacked bar
#     [
#         ("C", 1.2, "#1abc9c"),
#         ("Component B", 1.3, "#3498db"),
#         ("Component C", 1.1, "#e67e22"),
#     ],
# ]

# comparison_data_4bars = [
#     ("Current vs 2017", 1.677, "#e74c3c"),
#     ("Ho et Al", 2.5, "#f39c12"),
# ]

# create_stacked_comparison_plot(
#     stacked_groups_4bars,
#     comparison_data_4bars,
#     "Example: 4 Total Bars (2 Stacked + 2 Non-Stacked)",
# )

# %%
# Example: Custom bar ordering
# You can now specify the exact order of bars on the x-axis

# # Define your data
# stacked_groups_custom = [
#     [("Rotary Encoding", 1.4, "#9b59b6"), ("SwiGLU", 1.1, "#e67e22")],
#     [("Component A", 1.2, "#1abc9c"), ("Component B", 1.15, "#3498db")],
# ]

# comparison_data_custom = [
#     ("Current vs 2017", 1.677, "#e74c3c"),
#     ("Ho et Al", 2.5, "#f39c12"),
# ]

# # Example 1: Default order (all stacked first, then all comparison)
# print("Default order:")
# create_stacked_comparison_plot(
#     stacked_groups_custom,
#     comparison_data_custom,
#     "Default Order: All Stacked First, Then All Comparison",
# )

# # Example 2: Custom order - mix stacked and comparison bars
# print("\nCustom order - alternating:")
# create_stacked_comparison_plot(
#     stacked_groups_custom,
#     comparison_data_custom,
#     "Custom Order: Alternating Stacked and Comparison Bars",
#     bar_order=["stacked_0", "comparison_0", "stacked_1", "comparison_1"],
# )

# # Example 3: Custom order - comparison bars first
# print("\nCustom order - comparison first:")
# create_stacked_comparison_plot(
#     stacked_groups_custom,
#     comparison_data_custom,
#     "Custom Order: Comparison Bars First",
#     bar_order=["comparison_0", "comparison_1", "stacked_0", "stacked_1"],
# )

# %%
# # Example 4: 5 stacked components vs 3 comparisons
# components_data_3 = [
#     ("Optimizer", 2.8, '#e74c3c'),
#     ("Architecture", 2.2, '#3498db'),
#     ("Initialization", 1.6, '#2ECC71'),
#     ("Regularization", 1.3, '#f39c12'),
#     ("Data Augmentation", 1.2, '#9b59b6')
# ]

# comparison_data_3 = [
#     ("Baseline", 1.0, '#95a5a6'),
#     ("Partial Improvement", 4.5, '#e67e22'),
#     ("Full Pipeline", 8.2, '#1abc9c')
# ]

# create_stacked_comparison_plot(
#     components_data_3,
#     comparison_data_3,
#     "Complete ML Pipeline Components"
# )


# %%
# Orthogonal

# %%
# Example usage of compute_multiplier_closest_approach
# This compares an empirical training curve to a power law fit

# Example power law parameters (you would get these from fitting your data)
# Power law form: Loss = E + A * C^(-alpha)
# E_example = 3.5  # Irreducible loss
# A_example = 1e14  # Amplitude
# alpha_example = 0.05  # Exponent

# Example CSV file with empirical training curve
# example_csv = "../experimental_data_folder/stanford_mult/64d_adam.csv"

# multiplier, details = compute_multiplier_closest_approach(
#     csv_file=example_csv,
#     E=E_example,
#     A=A_example,
#     alpha=alpha_example,
#     verbose=True
# )

# print(f"\nMultiplier at closest approach: {multiplier:.3f}x")
# print(f"Details: {details}")

# %%
# Example usage with two power laws
# Power Law 1: Loss = E1 + A1 * C^(-alpha1)
# Power Law 2: Loss = E2 + A2 * C^(-alpha2)

# E1, A1, alpha1 = 3.5, 1e14, 0.05  # Modern transformer
# E2, A2, alpha2 = 3.8, 2e14, 0.045  # Older transformer (less efficient)

# multiplier, details = compute_multiplier_closest_approach(
#     (E1, A1, alpha1),  # Power law A
#     (E2, A2, alpha2),  # Power law B
#     verbose=True,
#     compute_range=(1e15, 1e18),  # Optional: specify compute range
#     num_samples=2000  # Optional: number of samples for comparison
# )

# print(f"\nMultiplier at closest approach: {multiplier:.3f}x")
# print(f"At the closest approach point, Power Law 2 needs {multiplier:.3f}x compute vs Power Law 1")

# %%
