#!/usr/bin/env python3
"""
Helper script to run scaling law analysis on different datasets.
This provides an easy interface to analyze specific datasets or compare multiple ones.
"""

import sys
from pathlib import Path
from fit_hitchhikers_loss import (
    get_dataset_configurations,
    fit_validation_loss_from_pairs,
    print_fit_results,
)


def list_available_datasets():
    """List all available dataset configurations."""
    configurations = get_dataset_configurations()

    print("Available dataset configurations:")
    print("=" * 50)

    for i, (name, pairs) in enumerate(configurations.items(), 1):
        status = f"({len(pairs)} files)" if pairs else "(no files found)"
        print(f"{i:2d}. {name:<30} {status}")

    return configurations


def run_specific_dataset(dataset_name: str, ignore_first_percent: float = 0.0):
    """Run analysis on a specific dataset."""
    configurations = get_dataset_configurations()

    if dataset_name not in configurations:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print("\nAvailable datasets:")
        for name in configurations.keys():
            print(f"  - {name}")
        return False

    pairs = configurations[dataset_name]
    if not pairs:
        print(f"Error: No files found for dataset '{dataset_name}'.")
        return False

    print(f"Running analysis on '{dataset_name}' with {len(pairs)} files...")

    try:
        fit = fit_validation_loss_from_pairs(
            pairs,
            loss_column="validation_loss",
            use_tokens_column=True,
            ignore_first_percent=ignore_first_percent,
        )
        print_fit_results(dataset_name, fit)
        return True

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        return False


def run_multiple_datasets(dataset_names: list, ignore_first_percent: float = 0.0):
    """Run analysis on multiple specific datasets."""
    configurations = get_dataset_configurations()
    results = {}

    for dataset_name in dataset_names:
        if dataset_name not in configurations:
            print(f"Warning: Dataset '{dataset_name}' not found, skipping.")
            continue

        pairs = configurations[dataset_name]
        if not pairs:
            print(f"Warning: No files found for dataset '{dataset_name}', skipping.")
            continue

        print(f"\nProcessing {dataset_name} with {len(pairs)} files...")

        try:
            fit = fit_validation_loss_from_pairs(
                pairs,
                loss_column="validation_loss",
                use_tokens_column=True,
                ignore_first_percent=ignore_first_percent,
            )
            results[dataset_name] = fit
            print_fit_results(dataset_name, fit)

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(
            f"{'Dataset':<30} {'Alpha':<10} {'Beta':<10} {'Exp(E)':<12} {'Points':<8}"
        )
        print("-" * 80)
        for name, fit in results.items():
            print(
                f"{name:<30} {fit.alpha:<10.4f} {fit.beta:<10.4f} {fit.exp_E:<12.6f} {fit.num_points:<8}"
            )


def main():
    """Main entry point with command line interface."""
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python run_scaling_analysis.py list                    # List available datasets"
        )
        print(
            "  python run_scaling_analysis.py all [ignore_percent]    # Run all datasets"
        )
        print(
            "  python run_scaling_analysis.py <dataset_name> [ignore_percent]  # Run specific dataset"
        )
        print(
            "  python run_scaling_analysis.py <name1> <name2> ... [ignore_percent]  # Run multiple datasets"
        )
        print()
        print("ignore_percent: Percentage of data to ignore from the beginning (0.0-99.9)")
        print()
        list_available_datasets()
        return

    command = sys.argv[1].lower()
    
    # Parse ignore_first_percent if provided
    ignore_first_percent = 0.0
    if len(sys.argv) > 2:
        try:
            ignore_first_percent = float(sys.argv[-1])  # Last argument
            if ignore_first_percent < 0.0 or ignore_first_percent >= 100.0:
                print(f"Error: ignore_first_percent must be between 0.0 and 99.9, got {ignore_first_percent}")
                return
        except ValueError:
            # If last argument is not a number, it's probably a dataset name
            ignore_first_percent = 0.0

    if command == "list":
        list_available_datasets()

    elif command == "all":
        # Import and run the all datasets function
        from fit_hitchhikers_loss import run_all_dataset_analyses

        run_all_dataset_analyses(ignore_first_percent=ignore_first_percent)

    else:
        # Treat as dataset name(s)
        dataset_names = sys.argv[1:]
        
        # Remove ignore_first_percent from dataset names if it was parsed
        if len(dataset_names) > 0:
            try:
                float(dataset_names[-1])
                dataset_names = dataset_names[:-1]  # Remove the last element if it's a number
            except ValueError:
                pass  # Last element is not a number, keep it

        if len(dataset_names) == 1:
            run_specific_dataset(dataset_names[0], ignore_first_percent=ignore_first_percent)
        else:
            run_multiple_datasets(dataset_names, ignore_first_percent=ignore_first_percent)


if __name__ == "__main__":
    main()
