#!/usr/bin/env python3
"""
Test script to demonstrate the new extrapolation functionality
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from no_color_scale_extensive import TrainingCurveAnalyzer, IRREDUCIBLE_LOSS


def test_extrapolation():
    """Test the new extrapolation functionality"""

    # Initialize analyzer
    analyzer = TrainingCurveAnalyzer(
        irreducible_loss=IRREDUCIBLE_LOSS, use_theoretical_flops=False
    )

    # Add a few test experiments (you can modify these paths as needed)
    test_experiments = [
        {
            "name": "32d new scaling",
            "csv_path": "../experimental_data_folder/new_scaling/32d_new_scaling.csv",
            "marker": "o",
            "color": "tab:purple",
            "include_in_frontier": True,
            "class": "transformer",
            "hidden_dim": 32,
        },
        {
            "name": "48d new scaling",
            "csv_path": "../experimental_data_folder/new_scaling/48d_new_scaling.csv",
            "marker": "o",
            "color": "tab:purple",
            "include_in_frontier": True,
            "class": "transformer",
            "hidden_dim": 48,
        },
        {
            "name": "64d new scaling",
            "csv_path": "../experimental_data_folder/new_scaling/64d_new_scaling.csv",
            "marker": "o",
            "color": "tab:purple",
            "include_in_frontier": True,
            "class": "transformer",
            "hidden_dim": 64,
        },
    ]

    # Add experiments
    for config in test_experiments:
        try:
            analyzer.add_experiment(
                name=config["name"],
                csv_path=config["csv_path"],
                color=config.get("color"),
                marker=config.get("marker", "o"),
                include_in_frontier=config.get("include_in_frontier", True),
                class_name=config.get("class"),
                hidden_dim=config.get("hidden_dim"),
            )
        except Exception as e:
            print(f"Could not load {config['name']}: {e}")

    if not analyzer.experiments:
        print("No experiments loaded. Please check the CSV file paths.")
        return

    print(f"Loaded {len(analyzer.experiments)} experiments")

    # Test different extrapolation factors
    extrapolation_factors = [1.5, 2.0, 3.0, 5.0]

    for factor in extrapolation_factors:
        print(f"\n=== Testing extrapolation factor: {factor} ===")

        # Plot with different extrapolation factors
        analyzer.plot_training_curves_by_class(
            show_all_curves=True,
            show_power_law_fit=True,
            show_sklearn_fit=False,
            classes_to_plot=["transformer"],
            extrapolation_factor=factor,
            save_path=f"test_extrapolation_{factor}.png",
        )

        print(f"Plot saved with extrapolation factor {factor}")


if __name__ == "__main__":
    test_extrapolation()
