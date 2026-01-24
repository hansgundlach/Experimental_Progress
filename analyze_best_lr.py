#!/usr/bin/env python3
"""
Analyze learning rate sweeps to find best learning rates for each dimension.
Reads through lr sweep folders and finds the best learning rate for each model dimension.
"""

import os
import pandas as pd
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np


def extract_dimension_and_lr(folder_name: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Extract dimension and learning rate from folder name.

    Examples:
    - "32d_sgd_experiment_lr_01.csv" -> (32, 0.1)
    - "vanilla_96d_lr_10em03.csv" -> (96, 0.001)
    - "64d_test_experiment_lr_001.csv" -> (64, 0.001)
    """
    # Remove .csv extension first
    base_name = folder_name.replace(".csv", "")

    # Extract dimension (look for number followed by 'd')
    dim_match = re.search(r"(\d+)d", base_name)
    dimension = int(dim_match.group(1)) if dim_match else None

    # Extract learning rate after 'lr_'
    lr_match = re.search(
        r"lr_([^_/]+)$", base_name
    )  # $ ensures we get to end of string
    if not lr_match:
        return dimension, None

    lr_str = lr_match.group(1)

    # Parse different learning rate formats
    try:
        # Handle NEW clear scientific notation: 10e-1, 10e-2, 3.2e-3, etc.
        if "e" in lr_str:
            # Check for custom format: XYeN means X.Y * 10^(-N) (e.g., 18e2 = 1.8e-2 = 0.018)
            # This format uses 2 digits before 'e' without +/- signs
            custom_format_match = re.match(r"^(\d{2})e(\d+)$", lr_str)
            if custom_format_match:
                # Format like "18e2" means 1.8 * 10^(-2) = 0.018
                # Format like "32e3" means 3.2 * 10^(-3) = 0.0032
                # Format like "56e4" means 5.6 * 10^(-4) = 0.00056
                mantissa = int(custom_format_match.group(1))
                exponent = int(custom_format_match.group(2))
                lr = (mantissa / 10.0) * (10 ** (-exponent))
            # Standard scientific notation with explicit +/- signs
            elif "+" in lr_str or "-" in lr_str:
                lr = float(lr_str)
            else:
                # Fallback: try standard scientific notation
                lr = float(lr_str)
        # Handle OLD confusing formats for backwards compatibility
        elif "em" in lr_str:
            lr_str = lr_str.replace("em", "e-")
            lr = float(lr_str)
        elif "ep" in lr_str:
            lr_str = lr_str.replace("ep", "e+")
            lr = float(lr_str)
        else:
            # Handle legacy specific formats for backwards compatibility
            lr_mapping = {
                "1": 1.0,
                "01": 0.1,
                "001": 0.001,
                "003": 0.003,  # Close to 10**(-2.5) ≈ 0.003162
                "0032": 0.003162,  # 10**(-2.5) ≈ 0.003162
                "32em03": 0.00032,  # 32e-03 = 32 * 10^(-3)
                "10em03": 0.001,  # 10e-03 = 10 * 10^(-3)
                "0001": 0.0001,
            }

            if lr_str in lr_mapping:
                lr = lr_mapping[lr_str]
            elif lr_str.isdigit() and len(lr_str) <= 4:
                # For other digit-only strings, try to parse as decimal
                if len(lr_str) == 1:
                    lr = float(lr_str)
                elif len(lr_str) == 2 and lr_str.startswith("0"):
                    lr = float("0." + lr_str[1])  # "01" -> 0.1
                elif len(lr_str) >= 3 and lr_str.startswith("00"):
                    lr = float("0.00" + lr_str[2:])  # "001" -> 0.001
                else:
                    lr = float(lr_str) / (10 ** (len(lr_str) - 1))
            else:
                # Direct float conversion for regular floats
                lr = float(lr_str)

    except ValueError:
        print(
            f"Warning: Could not parse learning rate '{lr_str}' from folder '{folder_name}'"
        )
        return dimension, None

    return dimension, lr


def get_final_validation_loss(csv_path: str) -> Optional[float]:
    """Get the final validation loss from a CSV file."""
    try:
        df = pd.read_csv(csv_path)

        # Look for validation loss columns (try different possible names)
        val_loss_cols = [
            col
            for col in df.columns
            if "validation" in col.lower() and "loss" in col.lower()
        ]
        if not val_loss_cols:
            val_loss_cols = [col for col in df.columns if "val_loss" in col.lower()]

        if not val_loss_cols:
            print(f"Warning: No validation loss column found in {csv_path}")
            return None

        val_loss_col = val_loss_cols[0]
        final_loss = df[val_loss_col].iloc[-1]

        return float(final_loss)

    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}")
        return None


def convert_lr_to_scientific(lr: float) -> str:
    """Convert learning rate to scientific notation format like 10**(-2.5)."""
    if lr == 0:
        return "0"

    log_lr = np.log10(lr)

    # Check if it's a nice power of 10
    if abs(log_lr - round(log_lr)) < 1e-10:
        exponent = int(round(log_lr))
        if exponent == 0:
            return "1"
        elif exponent > 0:
            return f"10**{exponent}"
        else:
            return f"10**({exponent})"
    else:
        # For non-integer exponents
        return f"10**({log_lr:.1f})"


def analyze_lr_sweeps_by_folder(
    base_folder: str = "experimental_data_folder",
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Analyze learning rate sweeps to find best learning rates for each dimension within each folder.

    Args:
        base_folder: Base folder containing experimental data

    Returns:
        Dictionary mapping folder_name to {dimension: (best_lr, best_validation_loss)}
    """
    if not os.path.exists(base_folder):
        print(f"Error: Base folder '{base_folder}' does not exist")
        return {}

    # Dictionary to store results: {folder_name: {dimension: (best_lr, best_loss)}}
    folder_results = {}

    # Find all CSV files that look like learning rate sweeps
    lr_files = []
    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if (
                file_name.endswith(".csv")
                and "lr_" in file_name
                and any(c.isdigit() for c in file_name)
            ):
                lr_files.append(os.path.join(root, file_name))

    print(f"Found {len(lr_files)} potential learning rate experiment files")

    processed_files = 0
    for file_path in lr_files:
        file_name = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        dimension, lr = extract_dimension_and_lr(file_name)

        if dimension is None or lr is None:
            continue

        final_loss = get_final_validation_loss(file_path)
        if final_loss is None or np.isnan(final_loss):
            continue

        processed_files += 1

        # Initialize folder if not exists
        if folder_name not in folder_results:
            folder_results[folder_name] = {}

        # Check if this is the best result for this dimension in this folder
        if (
            dimension not in folder_results[folder_name]
            or final_loss < folder_results[folder_name][dimension][1]
        ):
            folder_results[folder_name][dimension] = (lr, final_loss)

    print(
        f"\nProcessed {processed_files} learning rate experiment files across {len(folder_results)} folders"
    )
    return folder_results


def print_best_learning_rates_by_folder(
    folder_results: Dict[str, Dict[int, Tuple[float, float]]],
) -> None:
    """Print the best learning rates for each dimension within each folder."""
    if not folder_results:
        print("No results found!")
        return

    # Get all unique dimensions across all folders
    all_dimensions = set()
    for folder_data in folder_results.values():
        all_dimensions.update(folder_data.keys())
    all_dimensions = sorted(all_dimensions)

    print("\n" + "=" * 100)
    print("BEST LEARNING RATES BY DIMENSION FOR EACH FOLDER")
    print("=" * 100)

    # Print header
    header = f"{'Folder':<35}"
    for dim in all_dimensions:
        header += f"{str(dim)+'d':<12}"
    print(header)
    print("-" * 100)

    # Print results for each folder
    for folder_name in sorted(folder_results.keys()):
        folder_data = folder_results[folder_name]

        # Truncate folder name if too long
        display_folder = (
            folder_name[:32] + "..." if len(folder_name) > 32 else folder_name
        )
        row = f"{display_folder:<35}"

        for dim in all_dimensions:
            if dim in folder_data:
                best_lr, best_loss = folder_data[dim]
                scientific_lr = convert_lr_to_scientific(best_lr)
                row += f"{scientific_lr:<12}"
            else:
                row += f"{'--':<12}"

        print(row)

    print("-" * 100)
    print(f"Total folders analyzed: {len(folder_results)}")
    print(f"Dimensions found: {', '.join([str(d)+'d' for d in all_dimensions])}")
    print("=" * 100)


def print_detailed_folder_results(
    folder_results: Dict[str, Dict[int, Tuple[float, float]]],
) -> None:
    """Print detailed results for each folder separately."""
    if not folder_results:
        print("No results found!")
        return

    for folder_name in sorted(folder_results.keys()):
        folder_data = folder_results[folder_name]

        print(f"\n{'='*80}")
        print(f"FOLDER: {folder_name}")
        print("=" * 80)
        print(
            f"{'Dimension':<10} {'Best LR':<15} {'Scientific':<15} {'Final Loss':<12}"
        )
        print("-" * 80)

        for dimension in sorted(folder_data.keys()):
            best_lr, best_loss = folder_data[dimension]
            scientific_lr = convert_lr_to_scientific(best_lr)

            print(
                f"{dimension}d{'':<7} {best_lr:<15.6f} {scientific_lr:<15} {best_loss:<12.4f}"
            )

        print("-" * 80)
        print(f"Dimensions in this folder: {len(folder_data)}")


def print_best_learning_rates(best_results: Dict[int, Tuple[float, float]]) -> None:
    """Print the best learning rates for each dimension in a formatted table."""
    if not best_results:
        print("No results found!")
        return

    print("\n" + "=" * 60)
    print("BEST LEARNING RATES BY DIMENSION")
    print("=" * 60)
    print(f"{'Dimension':<10} {'Best LR':<20} {'Scientific':<15} {'Final Loss':<12}")
    print("-" * 60)

    # Sort by dimension
    for dimension in sorted(best_results.keys()):
        best_lr, best_loss = best_results[dimension]
        scientific_lr = convert_lr_to_scientific(best_lr)

        print(
            f"{dimension:<10} {best_lr:<20.6f} {scientific_lr:<15} {best_loss:<12.4f}"
        )

    print("-" * 60)
    print(f"Total dimensions analyzed: {len(best_results)}")
    print("=" * 60)


def main():
    """Main function to analyze learning rate sweeps."""
    import argparse

    # Default experimental data folder - change this easily
    DEFAULT_EXPERIMENTAL_FOLDER = "experimental_data_folder"

    parser = argparse.ArgumentParser(
        description="Analyze learning rate sweeps to find best LRs by folder"
    )
    parser.add_argument(
        "--folder",
        "-f",
        default=DEFAULT_EXPERIMENTAL_FOLDER,
        help=f"Base folder containing experimental data (default: {DEFAULT_EXPERIMENTAL_FOLDER})",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed results for each folder separately",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during processing",
    )

    args = parser.parse_args()

    print(f"Analyzing learning rate sweeps in: {args.folder}")

    # Analyze the learning rate sweeps by folder
    folder_results = analyze_lr_sweeps_by_folder(args.folder)

    # Print the results
    if args.detailed:
        print_detailed_folder_results(folder_results)
    else:
        print_best_learning_rates_by_folder(folder_results)

    if not folder_results:
        print("\nTroubleshooting tips:")
        print("1. Make sure the folder path is correct")
        print("2. Check that CSV files exist in the lr experiment folders")
        print("3. Verify that CSV files contain 'validation_loss' or similar columns")
        print("4. Folder names should contain patterns like '32d' and 'lr_001'")


if __name__ == "__main__":
    main()
