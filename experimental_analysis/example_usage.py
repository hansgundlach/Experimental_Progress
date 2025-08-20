# %%
"""
Example usage of the TrainingCurveAnalyzer class.
This script demonstrates how to load arbitrary CSV files and identify frontier points.
"""

from lstm_scaling import TrainingCurveAnalyzer
import matplotlib.pyplot as plt


def main():
    # Initialize the analyzer
    analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

    # Example 1: Add experiments with custom configuration
    # You can modify these paths to point to your actual CSV files
    experiments = [
        {
            "name": "16d No Rotary",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling_No_Rotary_123/16d_no_rotary_123.csv",
            "color": "tab:orange",
            "marker": "o",
            "linestyle": "-",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "24d No Rotary",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling_No_Rotary_123/24d_no_rotary_123.csv",
            "color": "tab:purple",
            "marker": "s",
            "linestyle": "-",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "32d No Rotary",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling_No_Rotary_123/32d_no_rotary_123.csv",
            "color": "tab:green",
            "marker": "^",
            "linestyle": "-",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "64d No Rotary",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling_No_Rotary_123/64d_no_rotary_123.csv",
            "color": "tab:blue",
            "marker": "D",
            "linestyle": "-",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "96d No Rotary",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling_No_Rotary_123/96d_no_rotary_123.csv",
            "color": "tab:red",
            "marker": "P",
            "linestyle": "-",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        # Example of an experiment that won't be included in frontier analysis
        # {
        #     "name": "Baseline Experiment",
        #     "csv_path": "../experimental_data_folder/baseline/baseline.csv",
        #     "color": "gray",
        #     "marker": "x",
        #     "linestyle": "--",
        #     "include_in_frontier": False,  # Won't be included in frontier analysis
        # },
    ]

    # Add all experiments
    for exp in experiments:
        analyzer.add_experiment(
            name=exp["name"],
            csv_path=exp["csv_path"],
            color=exp["color"],
            marker=exp["marker"],
            linestyle=exp["linestyle"],
            include_in_frontier=exp.get("include_in_frontier", True),
        )

    # Example 2: Identify frontier using Pareto method
    print("\n=== Identifying Frontier (Pareto Method) ===")
    frontier_pareto = analyzer.identify_frontier(method="pareto")

    # Example 3: Fit power law to frontier points
    print("\n=== Power Law Fit ===")
    power_law_result = analyzer.fit_power_law()
    if power_law_result:
        a, b = power_law_result
        print(f"Power law parameters: a = {a:.4e}, b = {b:.4f}")

    # Example 4: Plot all training curves with frontier highlighted
    print("\n=== Plotting Results ===")
    analyzer.plot_training_curves(
        show_all_curves=True,
        show_frontier_only=False,
        show_power_law_fit=True,
        save_path="Figures/training_curves_example.png",
    )

    # Example 5: Plot only frontier experiments
    print("\n=== Plotting Frontier Only ===")
    analyzer.plot_training_curves(
        show_all_curves=False,
        show_frontier_only=True,
        show_power_law_fit=True,
        save_path="Figures/frontier_only_example.png",
    )


def add_custom_experiments():
    """
    Example of how to add experiments with different column names or configurations.
    """
    analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

    # Example with custom column names
    analyzer.add_experiment(
        name="Custom Experiment 1",
        csv_path="path/to/your/experiment1.csv",
        compute_col="flops",  # Custom compute column name
        loss_col="val_loss",  # Custom loss column name
        color="red",
        marker="*",
        linestyle="--",
        alpha=0.8,
        include_in_frontier=True,  # Include in frontier analysis
    )

    # Example with auto-assigned color
    analyzer.add_experiment(
        name="Custom Experiment 2",
        csv_path="path/to/your/experiment2.csv",
        # Color will be auto-assigned from the palette
        marker="^",
        linestyle=":",
        include_in_frontier=True,  # Include in frontier analysis
    )

    return analyzer


def demonstrate_frontier_exclusion():
    """
    Example demonstrating how to exclude certain experiments from frontier analysis.
    """
    analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

    # Add experiments that will be included in frontier analysis
    analyzer.add_experiment(
        name="Main Experiment 1",
        csv_path="path/to/main1.csv",
        color="blue",
        marker="o",
        include_in_frontier=True,  # Will be considered for frontier
    )

    analyzer.add_experiment(
        name="Main Experiment 2",
        csv_path="path/to/main2.csv",
        color="red",
        marker="s",
        include_in_frontier=True,  # Will be considered for frontier
    )

    # Add experiments that will NOT be included in frontier analysis
    analyzer.add_experiment(
        name="Baseline Experiment",
        csv_path="path/to/baseline.csv",
        color="gray",
        marker="x",
        linestyle="--",
        include_in_frontier=False,  # Won't be considered for frontier
    )

    analyzer.add_experiment(
        name="Ablation Study",
        csv_path="path/to/ablation.csv",
        color="orange",
        marker="^",
        linestyle=":",
        include_in_frontier=False,  # Won't be considered for frontier
    )

    # Identify frontier (only considers experiments with include_in_frontier=True)
    frontier = analyzer.identify_frontier(method="pareto")
    print(f"Frontier experiments: {frontier}")

    # Plot all experiments (both frontier and non-frontier will be shown)
    analyzer.plot_training_curves(
        show_all_curves=True,
        show_frontier_only=False,
        show_power_law_fit=True,
        save_path="Figures/frontier_exclusion_example.png",
    )

    return analyzer


if __name__ == "__main__":
    main()

    # Uncomment to see custom experiment example
    # custom_analyzer = add_custom_experiments()

    # Uncomment to see frontier exclusion example
    # frontier_exclusion_analyzer = demonstrate_frontier_exclusion()

# %%
