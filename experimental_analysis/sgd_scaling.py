# %%
print("hello")

#%%
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Optional
import warnings


class TrainingCurveAnalyzer:
    """
    A class to analyze and plot training curves from multiple CSV files,
    identifying frontier points and fitting power laws.
    """

    def __init__(self, irreducible_loss: float = 1.76):
        """
        Initialize the analyzer.

        Args:
            irreducible_loss: The irreducible loss to subtract from validation losses
        """
        self.irreducible_loss = irreducible_loss
        self.experiments = {}
        self.frontier_points = []
        self.color_palette = list(mcolors.TABLEAU_COLORS.values())

    def add_experiment(
        self,
        name: str,
        csv_path: str,
        compute_col: str = "total_flops_profiler",
        loss_col: str = "validation_loss",
        color: Optional[str] = None,
        marker: str = "o",
        linestyle: str = "-",
        alpha: float = 0.6,
        include_in_frontier: bool = True,
    ) -> None:
        """
        Add an experiment from a CSV file.

        Args:
            name: Name for the experiment
            csv_path: Path to the CSV file
            compute_col: Column name for compute values
            loss_col: Column name for loss values
            color: Color for plotting (auto-assigned if None)
            marker: Marker style for plotting
            linestyle: Line style for plotting
            alpha: Transparency for plotting
            include_in_frontier: Whether to include this experiment in frontier analysis
        """
        try:
            df = pd.read_csv(csv_path)
            if compute_col not in df.columns:
                raise ValueError(f"Column '{compute_col}' not found in {csv_path}")
            if loss_col not in df.columns:
                raise ValueError(f"Column '{loss_col}' not found in {csv_path}")

            # Get final point
            final_point = df.iloc[-1]
            final_compute = final_point[compute_col]
            final_loss = final_point[loss_col] - self.irreducible_loss

            # Assign color if not provided
            if color is None:
                color = self.color_palette[
                    len(self.experiments) % len(self.color_palette)
                ]

            self.experiments[name] = {
                "data": df,
                "compute_col": compute_col,
                "loss_col": loss_col,
                "color": color,
                "marker": marker,
                "linestyle": linestyle,
                "alpha": alpha,
                "final_compute": final_compute,
                "final_loss": final_loss,
                "include_in_frontier": include_in_frontier,
            }

            print(
                f"Added experiment '{name}': final loss = {final_loss:.4f}, compute = {final_compute:.2e}"
            )

        except Exception as e:
            print(f"Error loading experiment '{name}' from {csv_path}: {e}")

    def identify_frontier(self, method: str = "pareto") -> List[str]:
        """
        Identify frontier points (best performing experiments).

        Args:
            method: Method to identify frontier ('pareto' or 'top_n')

        Returns:
            List of experiment names that are on the frontier
        """
        if not self.experiments:
            return []

        # Get all final points (only from experiments that should be included in frontier)
        points = []
        for name, exp in self.experiments.items():
            if exp["include_in_frontier"]:
                points.append((name, exp["final_compute"], exp["final_loss"]))

        if method == "pareto":
            # Find Pareto frontier (no other point dominates)
            frontier = []
            for name, compute, loss in points:
                is_dominated = False
                for other_name, other_compute, other_loss in points:
                    if (other_compute <= compute and other_loss < loss) or (
                        other_compute < compute and other_loss <= loss
                    ):
                        is_dominated = True
                        break
                if not is_dominated:
                    frontier.append(name)

            # Sort by compute for better visualization
            frontier.sort(key=lambda x: self.experiments[x]["final_compute"])

        elif method == "top_n":
            # Select top N experiments by loss (assuming lower is better)
            n = min(5, len(points))  # Default to top 5 or all if fewer
            frontier = sorted(points, key=lambda x: x[2])[:n]
            frontier = [name for name, _, _ in frontier]

        self.frontier_points = frontier
        print(f"Frontier experiments: {frontier}")
        return frontier

    def fit_power_law(
        self, experiment_names: Optional[List[str]] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Fit a power law to the specified experiments.

        Args:
            experiment_names: List of experiment names to fit (uses frontier if None)

        Returns:
            Tuple of (a, b) parameters for power law y = a * x^b, or None if fitting fails
        """
        if experiment_names is None:
            experiment_names = self.frontier_points

        if not experiment_names:
            print("No experiments to fit power law to")
            return None

        # Collect data points
        x_data = []
        y_data = []
        for name in experiment_names:
            if name in self.experiments:
                exp = self.experiments[name]
                # Only include in power law fit if it's marked for frontier analysis
                if exp["include_in_frontier"]:
                    x_data.append(exp["final_compute"])
                    y_data.append(exp["final_loss"])

        if len(x_data) < 2:
            print("Need at least 2 points to fit power law")
            return None

        # Fit power law
        def power_law(x, a, b):
            return a * np.power(x, b)

        try:
            params, covariance = curve_fit(power_law, x_data, y_data)
            a, b = params
            print(f"Power law fit: y = {a:.4e} * x^({b:.4f})")
            return a, b
        except Exception as e:
            print(f"Error fitting power law: {e}")
            return None

    def plot_training_curves(
        self,
        show_all_curves: bool = True,
        show_frontier_only: bool = False,
        show_power_law_fit: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training curves for all experiments.

        Args:
            show_all_curves: Whether to show all training curves
            show_frontier_only: Whether to only show frontier experiments
            show_power_law_fit: Whether to show power law fit
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)

        # Determine which experiments to plot
        if show_frontier_only:
            experiments_to_plot = {
                name: exp
                for name, exp in self.experiments.items()
                if name in self.frontier_points
            }
        else:
            experiments_to_plot = self.experiments

        # Plot training curves
        for name, exp in experiments_to_plot.items():
            compute_vals = exp["data"][exp["compute_col"]].values
            loss_vals = exp["data"][exp["loss_col"]].values - self.irreducible_loss

            # Only plot points with positive loss after subtraction
            mask = loss_vals > 0
            if np.any(mask):
                plt.plot(
                    compute_vals[mask],
                    loss_vals[mask],
                    marker=exp["marker"],
                    linestyle=exp["linestyle"],
                    color=exp["color"],
                    alpha=exp["alpha"],
                    label=f"{name} (training)",
                    linewidth=1,
                    markersize=3,
                )

        # Plot final points
        for name, exp in experiments_to_plot.items():
            is_frontier = name in self.frontier_points
            marker_size = 300 if is_frontier else 100
            marker = "*" if is_frontier else "o"
            zorder = 100 if is_frontier else 50

            plt.scatter(
                exp["final_compute"],
                exp["final_loss"],
                color=exp["color"],
                s=marker_size,
                marker=marker,
                label=f"{name} (final)",
                zorder=zorder,
                edgecolors="black" if is_frontier else None,
                linewidth=2 if is_frontier else 1,
            )

        # Plot power law fit if requested
        if show_power_law_fit and self.frontier_points:
            power_law_result = self.fit_power_law()
            if power_law_result is not None:
                a, b = power_law_result
                # Generate fit curve
                min_compute = min(
                    exp["final_compute"] for exp in experiments_to_plot.values()
                )
                max_compute = max(
                    exp["final_compute"] for exp in experiments_to_plot.values()
                )
                x_fit = np.logspace(np.log10(min_compute), np.log10(max_compute), 100)
                y_fit = a * np.power(x_fit, b)

                plt.plot(
                    x_fit,
                    y_fit,
                    "k--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Power law fit: y = {a:.2e} * x^({b:.4f})",
                )

        # Customize plot
        plt.xlabel("Compute (FLOPS)", fontsize=12)
        plt.ylabel("Validation Loss (Irreducible)", fontsize=12)
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.title(
            "Training Curves and Scaling Analysis", fontsize=14, fontweight="bold"
        )

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

    # Add experiments - you can modify these paths and names as needed
    experiments_config = [
        {
            "name": "16d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/16d_mup_sgd.csv",
            "color": "tab:orange",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "24d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/24d_mup_sgd.csv",
            "color": "tab:blue",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "32d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/32d_mup_sgd.csv",
            "color": "tab:green",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "48d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/48d_mup_sgd.csv",
            "color": "tab:red",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "64d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/64d_mup_sgd.csv",
            "color": "cyan",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "80d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/80d_2_mup_sgd.csv",
            "color": "tab:orange",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "96d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/96d_2_mup_sgd.csv",
            "color": "tab:red",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "128d sgd",
            "csv_path": "../experimental_data_folder/muP_scaling_experiments/128d_2_mup_sgd.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
        },
        {
            "name": "adam 16d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/16d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        {
            "name": "adam 24d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/24d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        {
            "name": "adam 32d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/32d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        {
            "name": "adam 64d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/64d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        {
            "name": "adam 96d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/96d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        {
            "name": "adam 128d",
            "csv_path": "../experimental_data_folder/Hidden_Dim_Scaling/128d_123.csv",
            "color": "tab:purple",
            "marker": "o",
            "include_in_frontier": False,  # Include in frontier analysis
        },
        # Example of an experiment that won't be included in frontier analysis
        # {
        #     "name": "baseline experiment",
        #     "csv_path": "../experimental_data_folder/baseline/baseline.csv",
        #     "color": "gray",
        #     "marker": "s",
        #     "include_in_frontier": False,  # Won't be included in frontier analysis
        # },
    ]

    # Add experiments
    for config in experiments_config:
        analyzer.add_experiment(
            name=config["name"],
            csv_path=config["csv_path"],
            color=config.get("color"),
            marker=config.get("marker", "o"),
            include_in_frontier=config.get("include_in_frontier", True),
        )

    # Identify frontier
    analyzer.identify_frontier(method="pareto")

    # Plot results
    analyzer.plot_training_curves(
        show_all_curves=True,
        show_frontier_only=False,
        show_power_law_fit=True,
        save_path="Figures/universal_scaling_law_study.png",
    )

# %%
