# %%
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Optional
import warnings


IRREDUCIBLE_LOSS = 1.8


class TrainingCurveAnalyzer:
    """
    A class to analyze and plot training curves from multiple CSV files,
    identifying frontier points and fitting power laws.
    """

    def __init__(self, irreducible_loss: float = IRREDUCIBLE_LOSS):
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
            final_loss_raw = final_point[loss_col]  # Store raw loss too

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
                "final_loss_raw": final_loss_raw,  # Add raw loss
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
        self,
        experiment_names: Optional[List[str]] = None,
        fit_type: str = "irreducible_subtracted",
    ) -> Optional[Tuple[float, ...]]:
        """
        Fit a power law to the specified experiments.

        Args:
            experiment_names: List of experiment names to fit (uses frontier if None)
            fit_type: Type of fit to perform:
                - "irreducible_subtracted": Fit (loss - irreducible) = A * C^alpha
                - "full_model": Fit loss = E + A * C^alpha (3-parameter fit)

        Returns:
            - For "irreducible_subtracted": Tuple of (a, b) parameters for power law y = a * x^b
            - For "full_model": Tuple of (E, A, alpha) parameters for y = E + A * x^alpha
            - None if fitting fails
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
                    if fit_type == "irreducible_subtracted":
                        y_data.append(
                            exp["final_loss"]
                        )  # Already has irreducible subtracted
                    elif fit_type == "full_model":
                        # Use raw loss (add back the irreducible loss)
                        y_data.append(exp["final_loss"] + self.irreducible_loss)

        if len(x_data) < 2:
            print("Need at least 2 points to fit power law")
            return None

        if fit_type == "irreducible_subtracted":
            # Original fit: (loss - irreducible) = A * C^alpha
            def power_law(x, a, b):
                return a * np.power(x, b)

            try:
                params, covariance = curve_fit(power_law, x_data, y_data)
                a, b = params
                print(
                    f"Power law fit (irreducible subtracted): y = {a:.4e} * x^({b:.4f})"
                )
                return a, b
            except Exception as e:
                print(f"Error fitting power law: {e}")
                return None

        elif fit_type == "full_model":
            # New fit: loss = exp(E) + exp(A) * C^alpha (exponential form for guaranteed positivity)
            def full_scaling_law(x, E_log, A_log, alpha):
                return np.exp(E_log) + np.exp(A_log) * np.power(x, alpha)

            try:
                # Initial guesses in log space for exp(E) + exp(A) * C^alpha
                E_baseline = max(np.percentile(y_data, 10), 1e-6)
                E_log_init = np.log(E_baseline)  # Log of baseline loss

                range_y = max(
                    np.percentile(y_data, 90) - np.percentile(y_data, 10), 1e-3
                )
                A_log_init = np.log(range_y)  # Log of scaling coefficient

                alpha_init = -0.5  # Common scaling exponent (negative)

                initial_guess = [E_log_init, A_log_init, alpha_init]

                # No bounds needed since exp() guarantees positivity for E and A
                # Only constrain alpha to be negative
                lower_bounds = [-np.inf, -np.inf, -np.inf]
                upper_bounds = [np.inf, np.inf, -0.001]  # alpha < 0

                params, covariance = curve_fit(
                    full_scaling_law,
                    x_data,
                    y_data,
                    p0=initial_guess,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000,  # Increase max iterations
                )
                E_log, A_log, alpha = params

                # Convert to exponential form for display
                E_exp = np.exp(E_log)
                A_exp = np.exp(A_log)

                # Calculate R-squared for goodness of fit
                y_pred = full_scaling_law(np.array(x_data), E_log, A_log, alpha)
                ss_res = np.sum((np.array(y_data) - y_pred) ** 2)
                ss_tot = np.sum((np.array(y_data) - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                print(
                    f"Full scaling law fit: y = {E_exp:.4f} + {A_exp:.4e} * x^({alpha:.4f})"
                )
                print(f"R-squared: {r_squared:.4f}")
                print(
                    f"Fitted irreducible loss: {E_exp:.4f} (original assumption: {self.irreducible_loss:.4f})"
                )
                print(f"Log-space parameters: E_log={E_log:.4f}, A_log={A_log:.4f}")

                # Return exponential values for consistency with previous interface
                return E_exp, A_exp, alpha
            except Exception as e:
                print(f"Error fitting full scaling law: {e}")
                return None

        else:
            raise ValueError(
                f"Unknown fit_type: {fit_type}. Use 'irreducible_subtracted' or 'full_model'"
            )

    def plot_training_curves(
        self,
        show_all_curves: bool = True,
        show_frontier_only: bool = False,
        show_power_law_fit: bool = True,
        fit_type: str = "irreducible_subtracted",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training curves for all experiments.

        Args:
            show_all_curves: Whether to show all training curves
            show_frontier_only: Whether to only show frontier experiments
            show_power_law_fit: Whether to show power law fit
            fit_type: Type of fit to show ("irreducible_subtracted" or "full_model")
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

            if fit_type == "irreducible_subtracted":
                loss_vals = exp["data"][exp["loss_col"]].values - self.irreducible_loss
                # Only plot points with positive loss after subtraction
                mask = loss_vals > 0
            else:  # full_model
                loss_vals = exp["data"][exp["loss_col"]].values  # Use raw loss
                mask = np.ones(len(loss_vals), dtype=bool)  # Plot all points

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

            # Use appropriate final loss value based on fit type
            if fit_type == "irreducible_subtracted":
                final_loss = exp["final_loss"]
            else:  # full_model
                final_loss = exp["final_loss_raw"]

            plt.scatter(
                exp["final_compute"],
                final_loss,
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
            power_law_result = self.fit_power_law(fit_type=fit_type)
            if power_law_result is not None:
                # Generate fit curve
                min_compute = min(
                    exp["final_compute"] for exp in experiments_to_plot.values()
                )
                max_compute = max(
                    exp["final_compute"] for exp in experiments_to_plot.values()
                )
                x_fit = np.logspace(np.log10(min_compute), np.log10(max_compute), 100)

                if fit_type == "irreducible_subtracted":
                    a, b = power_law_result
                    y_fit = a * np.power(x_fit, b)
                    fit_label = f"Power law fit: y = {a:.2e} * x^({b:.4f})"
                else:  # full_model
                    E_exp, A_exp, alpha = power_law_result
                    y_fit = E_exp + A_exp * np.power(x_fit, alpha)
                    fit_label = f"Scaling law fit: y = {E_exp:.4f} + {A_exp:.2e} * x^({alpha:.4f})"

                plt.plot(
                    x_fit,
                    y_fit,
                    "k--",
                    linewidth=2,
                    alpha=0.8,
                    label=fit_label,
                )

        # Customize plot
        plt.xlabel("Compute (FLOPS)", fontsize=12)
        if fit_type == "irreducible_subtracted":
            plt.ylabel("Validation Loss (Irreducible Subtracted)", fontsize=12)
        else:
            plt.ylabel("Validation Loss", fontsize=12)
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
    analyzer = TrainingCurveAnalyzer(irreducible_loss=IRREDUCIBLE_LOSS)

    # Add experiments - you can modify these paths and names as needed
    experiments_config = [
        # {
        #     "name": "56d no rotary",
        #     "csv_path": "../experimental_data_folder/vanilla_scaling_no_rotary/vanilla_56d_no_rot.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "vanilla_transformer_no_rotary",
        #     "hidden_dim": 56,
        # },
        # {
        #     "name": "64d no rotary",
        #     "csv_path": "../experimental_data_folder/vanilla_scaling_no_rotary/vanilla_64d_no_rot.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "vanilla_transformer_no_rotary",
        #     "hidden_dim": 64,
        # },
        # {
        #     "name": "32 vanilla optimal lr",
        #     "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_32d.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "vanilla_transformer",
        #     "hidden_dim": 32,
        # },
        # lstm scaling
        # {
        #     "name": "32d lstm optimal lr",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/32d_lstm_experiment.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "hidden_dim": 32,
        # },
        # {
        #     "name": "48d lstm optimal lr",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/48d_lstm_experiment.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "hidden_dim": 48,
        # },
        # {
        #     "name": "64d lstm optimal lr",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/64d_lstm_experiment.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "hidden_dim": 64,
        # },
        # # lr sweep lstm
        # {
        #     "name": "32d lstm lr sweep",
        #     "csv_path": "../experimental_data_folder/lstm_scaling_lr_sweep/32d_lstm_experiment_lr_10e2.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:red",
        #     "hidden_dim": 32,
        # },
        # # low drouput scaling diagnostic
        # {
        #     "name": "32d lstm low dropout scaling diagnostic",
        #     "csv_path": "../experimental_data_folder/lstm_scaling_diagnostic/32d_lstm_low_dropout.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:red",
        #     "hidden_dim": 32,
        # },
        # # tbptt scaling diagnostic
        # {
        #     "name": "32d lstm tbptt scaling diagnostic",
        #     "csv_path": "../experimental_data_folder/lstm_scaling_diagnostic/32d_lstm_no_dropout_002_warmup_testtbptt64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:purple",
        #     "hidden_dim": 32,
        # },
        # {
        #     "name": "32d lstm tbptt scaling diagnostic",
        #     "csv_path": "../experimental_data_folder/lstm_scaling_diagnostic/48d_batchsize64ll.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:purple",
        #     "hidden_dim": 32,
        # },
        # # lstm scaling
        # {
        #     "name": "32d lstm scaling",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/32d_lstm_scaling.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:cyan",
        #     "hidden_dim": 32,
        # },
        # {
        #     "name": "48d lstm scaling",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/48d_lstm_scaling.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:cyan",
        #     "hidden_dim": 48,
        # },
        # {
        #     "name": "64d lstm scaling",
        #     "csv_path": "../experimental_data_folder/lstm_scaling/64d_lstm_scaling.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:cyan",
        #     "hidden_dim": 64,
        # },
        # # lstm lr rate scaling experiment
        # {
        #     "name": "32d lstm lr rate scaling experiment",
        #     "csv_path": "../experimental_data_folder/lstm_scaling_lr_sweep/yy32d_lstm_scaling_bs64_lr_10e2.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "lstm",
        #     "color": "tab:cyan",
        #     "hidden_dim": 32,
        # },
        # lstm scaling experiments
        {
            "name": "32d lstm scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling/yy32d_lstm_scaling_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "lstm",
            "color": "deeppink",
            "hidden_dim": 32,
        },
        {
            "name": "48d lstm scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling/yy48d_lstm_scaling_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "lstm",
            "color": "deeppink",
            "hidden_dim": 48,
        },
        {
            "name": "64d lstm scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling/yy64d_lstm_scaling_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "lstm",
            "color": "deeppink",
            "hidden_dim": 64,
        },
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

    # Plot results with traditional fit (irreducible subtracted)
    analyzer.plot_training_curves(
        show_all_curves=True,
        show_frontier_only=False,
        show_power_law_fit=True,
        fit_type="irreducible_subtracted",
        save_path="Figures/universal_scaling_law_study.png",
    )

    # option for full_model or irreducible_subtracted

    # Plot results with full model fit (E + A*C^alpha)
    analyzer.plot_training_curves(
        show_all_curves=True,
        show_frontier_only=False,
        show_power_law_fit=True,
        fit_type="irreducible_subtracted",
        save_path="Figures/universal_scaling_law_study_full_model.png",
    )

# %
# %%
