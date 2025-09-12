# %%
print("hello world")
# %%
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
        # Per-class frontier storage
        self.frontier_points_by_class: Dict[str, List[str]] = {}
        self.frontier_points_all_by_class: Dict[str, List[Tuple[str, float, float]]] = (
            {}
        )

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
        class_name: Optional[str] = None,
        hidden_dim: Optional[int] = None,
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
            class_name: Class/category name for the experiment
            hidden_dim: Hidden dimension size for the model
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
                "class": class_name or "default",
                "hidden_dim": hidden_dim,
            }

            print(
                f"Added experiment '{name}': final loss = {final_loss:.4f}, compute = {final_compute:.2e}"
            )

        except Exception as e:
            print(f"Error loading experiment '{name}' from {csv_path}: {e}")

    def identify_frontier(
        self,
        method: str = "pareto",
        flop_range: Optional[Tuple[float, float]] = None,
        use_all_points: bool = True,
    ) -> List[str]:
        """
        Identify frontier points (best performing experiments).

        Args:
            method: Method to identify frontier ('pareto' or 'top_n')
            flop_range: Optional tuple of (min_flops, max_flops) to filter experiments/points
            use_all_points: If True, compute Pareto frontier over all training points

        Returns:
            List of experiment names that are on the frontier
        """
        if not self.experiments:
            return []

        # Assemble candidate points
        points: List[Tuple[str, float, float]] = []
        if use_all_points:
            # Use every row from each experiment's CSV
            for name, exp in self.experiments.items():
                if not exp["include_in_frontier"]:
                    continue
                df = exp["data"]
                comp_col = exp["compute_col"]
                loss_col = exp["loss_col"]
                for _, row in df.iterrows():
                    compute = float(row[comp_col])
                    loss = float(row[loss_col]) - self.irreducible_loss
                    if not np.isfinite(compute) or not np.isfinite(loss):
                        continue
                    if flop_range is not None:
                        min_flops, max_flops = flop_range
                        if compute < min_flops or compute > max_flops:
                            continue
                    # Only consider positive losses after subtraction
                    if loss <= 0:
                        continue
                    points.append((name, compute, loss))
        else:
            # Use only the final points per experiment
            for name, exp in self.experiments.items():
                if exp["include_in_frontier"]:
                    compute = exp["final_compute"]
                    # Filter by FLOP range if specified
                    if flop_range is not None:
                        min_flops, max_flops = flop_range
                        if compute < min_flops or compute > max_flops:
                            continue
                    points.append((name, compute, exp["final_loss"]))

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
                    # Keep the name so legacy callers still work
                    frontier.append(name)

            # Also store raw coordinates for frontier points when using all points
            if use_all_points:
                self.frontier_points_all = [
                    (n, c, l)
                    for (n, c, l) in points
                    if not any(
                        ((oc <= c and ol < l) or (oc < c and ol <= l))
                        for (_, oc, ol) in points
                    )
                ]

            # Sort by compute for better visualization
            frontier.sort(key=lambda x: self.experiments[x]["final_compute"])

        elif method == "top_n":
            # Select top N experiments by loss (assuming lower is better)
            n = min(5, len(points))  # Default to top 5 or all if fewer
            frontier = sorted(points, key=lambda x: x[2])[:n]
            frontier = [name for name, _, _ in frontier]

        self.frontier_points = frontier
        flop_range_str = f" in range {flop_range}" if flop_range else ""
        if use_all_points:
            count = len(getattr(self, "frontier_points_all", []))
            print(
                f"Frontier points{flop_range_str} (all training points): {count} points"
            )
        else:
            print(f"Frontier experiments{flop_range_str}: {frontier}")
        return frontier

    def identify_frontier_by_class(
        self,
        method: str = "pareto",
        flop_range: Optional[Tuple[float, float]] = None,
        use_all_points: bool = True,
        classes: Optional[List[str]] = None,
        flop_range_by_class: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Identify separate frontiers for each experiment class.

        Args:
            method: 'pareto' or 'top_n'
            flop_range: Optional (min_flops, max_flops) applied to all classes unless overridden
            use_all_points: If True, compute Pareto frontier over all training points
            classes: If provided, restrict computation to these classes
            flop_range_by_class: Optional per-class FLOP ranges; overrides flop_range for those classes

        Returns:
            Mapping from class name to list of experiment names on that class frontier
            (for all-points mode, this is the set of experiment names that contribute
            at least one frontier point).
        """
        if not self.experiments:
            return {}

        # Determine which classes to include
        if classes is None:
            classes = sorted(
                {exp.get("class", "default") for exp in self.experiments.values()}
            )

        # Group candidate points by class
        points_by_class: Dict[str, List[Tuple[str, float, float]]] = {}
        if use_all_points:
            for name, exp in self.experiments.items():
                if not exp["include_in_frontier"]:
                    continue
                cls = exp.get("class", "default")
                if cls not in classes:
                    continue
                df = exp["data"]
                comp_col = exp["compute_col"]
                loss_col = exp["loss_col"]
                for _, row in df.iterrows():
                    compute = float(row[comp_col])
                    loss = float(row[loss_col]) - self.irreducible_loss
                    if not np.isfinite(compute) or not np.isfinite(loss):
                        continue
                    # Choose applicable flop range for this class
                    active_range = None
                    if flop_range_by_class is not None and cls in flop_range_by_class:
                        active_range = flop_range_by_class[cls]
                    elif flop_range is not None:
                        active_range = flop_range
                    if active_range is not None:
                        min_flops, max_flops = active_range
                        if compute < min_flops or compute > max_flops:
                            continue
                    if loss <= 0:
                        continue
                    points_by_class.setdefault(cls, []).append((name, compute, loss))
        else:
            for name, exp in self.experiments.items():
                if not exp["include_in_frontier"]:
                    continue
                cls = exp.get("class", "default")
                if cls not in classes:
                    continue
                compute = float(exp["final_compute"])
                active_range = None
                if flop_range_by_class is not None and cls in flop_range_by_class:
                    active_range = flop_range_by_class[cls]
                elif flop_range is not None:
                    active_range = flop_range
                if active_range is not None:
                    min_flops, max_flops = active_range
                    if compute < min_flops or compute > max_flops:
                        continue
                loss = float(exp["final_loss"])
                points_by_class.setdefault(cls, []).append((name, compute, loss))

        frontier_names_by_class: Dict[str, List[str]] = {}
        frontier_points_all_by_class: Dict[str, List[Tuple[str, float, float]]] = {}

        for cls, points in points_by_class.items():
            if not points:
                frontier_names_by_class[cls] = []
                frontier_points_all_by_class[cls] = []
                continue
            if method == "pareto":
                # Compute Pareto frontier within class
                non_dominated = []
                for name, compute, loss in points:
                    is_dominated = False
                    for oname, ocompute, oloss in points:
                        if (ocompute <= compute and oloss < loss) or (
                            ocompute < compute and oloss <= loss
                        ):
                            is_dominated = True
                            break
                    if not is_dominated:
                        non_dominated.append((name, compute, loss))

                # Unique experiment names for frontier membership
                names = sorted({n for (n, _, _) in non_dominated})

                if use_all_points:
                    # Sort points for visualization by compute
                    non_dominated.sort(key=lambda t: t[1])
                    frontier_points_all_by_class[cls] = non_dominated
                else:
                    # Sort names by their final compute for consistency
                    names.sort(key=lambda n: self.experiments[n]["final_compute"])

                frontier_names_by_class[cls] = names
            else:  # top_n
                n = min(5, len(points))
                top = sorted(points, key=lambda x: x[2])[:n]
                names = [name for name, _, _ in top]
                frontier_names_by_class[cls] = names
                if use_all_points:
                    frontier_points_all_by_class[cls] = top

        self.frontier_points_by_class = frontier_names_by_class
        if use_all_points:
            self.frontier_points_all_by_class = frontier_points_all_by_class

        flop_range_str = ""
        if flop_range_by_class:
            flop_range_str = f" with per-class ranges {flop_range_by_class}"
        elif flop_range:
            flop_range_str = f" in range {flop_range}"
        if use_all_points:
            for cls, pts in self.frontier_points_all_by_class.items():
                print(
                    f"Class '{cls}' frontier points{flop_range_str} (all training points): {len(pts)} points"
                )
        else:
            for cls, names in self.frontier_points_by_class.items():
                print(f"Class '{cls}' frontier experiments{flop_range_str}: {names}")

        return frontier_names_by_class

    def fit_power_law_by_class(
        self,
        class_names: Optional[List[str]] = None,
        use_all_points: bool = True,
    ) -> Dict[str, Optional[Tuple[float, float, float]]]:
        """
        Fit power laws separately for each class in `class_names`.

        Args:
            class_names: Which classes to fit. Defaults to all present in stored frontiers.
            use_all_points: Whether to fit using all-point frontier or final-point frontier.

        Returns:
            Mapping from class name to (a, b, r_squared), or None if fit fails/insufficient points.
        """
        # Determine classes present
        if class_names is None:
            if use_all_points:
                class_names = list(self.frontier_points_all_by_class.keys())
            else:
                class_names = list(self.frontier_points_by_class.keys())

        results: Dict[str, Optional[Tuple[float, float, float]]] = {}

        def power_law(x, a, b):
            return a * np.power(x, b)

        for cls in class_names:
            if use_all_points:
                pts = self.frontier_points_all_by_class.get(cls, [])
                x_data = np.array([float(c) for (_, c, _) in pts])
                y_data = np.array([float(l) for (_, _, l) in pts])
            else:
                names = self.frontier_points_by_class.get(cls, [])
                x_data = np.array(
                    [float(self.experiments[n]["final_compute"]) for n in names]
                )
                y_data = np.array(
                    [float(self.experiments[n]["final_loss"]) for n in names]
                )

            if x_data.size < 2:
                print(f"Need at least 2 points to fit power law for class '{cls}'")
                results[cls] = None
                continue

            try:
                params, _ = curve_fit(power_law, x_data, y_data)
                a, b = params
                y_pred = power_law(x_data, a, b)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(
                    f"Class '{cls}' power law fit: y = {a:.4e} * x^({b:.4f}), R² = {r_squared:.4f}"
                )
                results[cls] = (a, b, r_squared)
            except Exception as e:
                print(f"Error fitting power law for class '{cls}': {e}")
                results[cls] = None

        return results

    def plot_training_curves_by_class(
        self,
        show_all_curves: bool = True,
        show_power_law_fit: bool = True,
        flop_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        use_all_points: bool = True,
        classes_to_plot: Optional[List[str]] = None,
        flop_range_by_class: Optional[Dict[str, Tuple[float, float]]] = None,
        colormap: str = "viridis",
    ) -> None:
        """
        Plot training curves and per-class frontiers and fits.
        Colors experiments by hidden dimension and shows experiment classes in legend.
        """
        # Optionally recompute per-class frontier for this plot only if a range is provided
        original_frontier_by_class = self.frontier_points_by_class.copy()
        original_frontier_all_by_class = getattr(
            self, "frontier_points_all_by_class", {}
        ).copy()
        if flop_range is not None or flop_range_by_class is not None:
            self.identify_frontier_by_class(
                method="pareto",
                flop_range=flop_range,
                use_all_points=use_all_points,
                classes=classes_to_plot,
                flop_range_by_class=flop_range_by_class,
            )

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Collect all hidden dimensions to set up colormap
        hidden_dims = []
        for exp in self.experiments.values():
            hidden_dim = exp.get("hidden_dim")
            if hidden_dim is not None:
                hidden_dims.append(hidden_dim)

        if hidden_dims:
            min_dim = min(hidden_dims)
            max_dim = max(hidden_dims)
            # Create colormap normalizer
            norm = plt.Normalize(vmin=min_dim, vmax=max_dim)
            cmap = plt.cm.get_cmap(colormap)

        # Filter classes to plot
        if classes_to_plot is None:
            classes_to_plot = sorted(
                {exp.get("class", "default") for exp in self.experiments.values()}
            )

        # Track which classes have been added to legend
        legend_classes = set()

        # Optionally plot all training curves (colored by hidden dimension)
        if show_all_curves:
            for name, exp in self.experiments.items():
                cls = exp.get("class", "default")
                if cls not in classes_to_plot:
                    continue

                # Get color based on hidden dimension
                hidden_dim = exp.get("hidden_dim")
                if hidden_dim is not None and hidden_dims:
                    color = cmap(norm(hidden_dim))
                else:
                    # Fallback to default color if no hidden_dim
                    color = exp.get("color", "tab:blue")

                compute_vals = exp["data"][exp["compute_col"]].values
                loss_vals = exp["data"][exp["loss_col"]].values - self.irreducible_loss
                mask = loss_vals > 0
                if np.any(mask):
                    # Only add class to legend once per class
                    label = cls if cls not in legend_classes else None
                    if label:
                        legend_classes.add(cls)

                    ax.plot(
                        compute_vals[mask],
                        loss_vals[mask],
                        marker=exp["marker"],
                        linestyle=exp["linestyle"],
                        color=color,
                        alpha=exp["alpha"],
                        label=label,
                        linewidth=2,
                        markersize=6,
                    )

        # Plot class-specific frontier points as stars (colored by hidden dimension)
        if use_all_points:
            for cls in classes_to_plot:
                pts = self.frontier_points_all_by_class.get(cls, [])
                for name, comp, loss in pts:
                    if name in self.experiments:
                        hidden_dim = self.experiments[name].get("hidden_dim")
                        if hidden_dim is not None and hidden_dims:
                            color = cmap(norm(hidden_dim))
                        else:
                            color = self.experiments[name].get("color", "tab:blue")
                    else:
                        color = "k"

                    ax.scatter(
                        comp,
                        loss,
                        color=color,
                        s=200,
                        marker="*",
                        zorder=120,
                        edgecolors="black",
                        linewidth=2,
                        label=None,
                    )
        else:
            for cls in classes_to_plot:
                names = self.frontier_points_by_class.get(cls, [])
                for name in names:
                    exp = self.experiments[name]
                    hidden_dim = exp.get("hidden_dim")
                    if hidden_dim is not None and hidden_dims:
                        color = cmap(norm(hidden_dim))
                    else:
                        color = exp.get("color", "tab:blue")

                    ax.scatter(
                        exp["final_compute"],
                        exp["final_loss"],
                        color=color,
                        s=400,
                        marker="*",
                        zorder=100,
                        edgecolors="black",
                        linewidth=2.5,
                        label=None,
                    )

        # Plot per-class power-law fits
        if show_power_law_fit:
            fit_results = self.fit_power_law_by_class(
                class_names=classes_to_plot, use_all_points=use_all_points
            )
            for cls, params in fit_results.items():
                if params is None:
                    continue
                a, b, r2 = params
                if use_all_points:
                    xs = [
                        comp
                        for (_, comp, _) in self.frontier_points_all_by_class.get(
                            cls, []
                        )
                    ]
                else:
                    xs = [
                        self.experiments[n]["final_compute"]
                        for n in self.frontier_points_by_class.get(cls, [])
                    ]
                if len(xs) < 2:
                    continue
                min_compute = min(xs)
                max_compute = max(xs)
                x_fit = np.logspace(np.log10(min_compute), np.log10(max_compute), 100)
                y_fit = a * np.power(x_fit, b)
                # Use a unique linestyle per class for clarity; color black to overlay
                linestyle = "--"
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle,
                    linewidth=3,
                    alpha=0.9,
                    label=f"{cls} fit: \n y = {a:.2e} * x^({b:.3f})",
                    color="black",
                )

        # Add colorbar for hidden dimension scale
        if hidden_dims:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # Create colorbar with explicit positioning
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8, aspect=20)
            cbar.set_label("Hidden Dimension", fontsize=16, rotation=270, labelpad=25)
            # Set the colorbar ticks to show actual hidden dimension values
            tick_values = sorted(list(set(hidden_dims)))
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([str(int(dim)) for dim in tick_values])
            # Increase colorbar tick label size
            cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel("Compute (FLOPS)", fontsize=18)
        ax.set_ylabel("Validation Loss (Irreducible)", fontsize=18)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "Per-Class Training Curves and Scaling Analysis",
            fontsize=20,
            fontweight="bold",
        )
        # Increase tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=14)

        # Position legend to avoid colorbar
        if hidden_dims:
            ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left", fontsize=16)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)

        # Adjust layout to accommodate colorbar and legend
        if hidden_dims:
            plt.subplots_adjust(right=0.7)
        else:
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        # Restore original per-class frontiers if we recalculated for this plot
        if flop_range is not None or flop_range_by_class is not None:
            self.frontier_points_by_class = original_frontier_by_class
            self.frontier_points_all_by_class = original_frontier_all_by_class

    def fit_power_law(
        self,
        experiment_names: Optional[List[str]] = None,
        use_all_points: bool = True,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fit a power law to the specified experiments.

        Args:
            experiment_names: List of experiment names to fit (uses frontier if None)

        Returns:
            Tuple of (a, b, r_squared) parameters for power law y = a * x^b, or None if fitting fails
        """
        # Collect data points from either the all-point frontier or final-point frontier
        x_data: List[float] = []
        y_data: List[float] = []
        if use_all_points:
            frontier_pts = getattr(self, "frontier_points_all", [])
            for _, comp, loss in frontier_pts:
                x_data.append(float(comp))
                y_data.append(float(loss))
        else:
            if experiment_names is None:
                experiment_names = self.frontier_points
            if not experiment_names:
                print("No experiments to fit power law to")
                return None
            for name in experiment_names:
                if name in self.experiments:
                    exp = self.experiments[name]
                    if exp["include_in_frontier"]:
                        x_data.append(exp["final_compute"])
                        y_data.append(exp["final_loss"])

        if len(x_data) < 2:
            print("Need at least 2 points to fit power law")
            return None

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Fit power law
        def power_law(x, a, b):
            return a * np.power(x, b)

        try:
            params, covariance = curve_fit(power_law, x_data, y_data)
            a, b = params

            # Calculate R-squared
            y_pred = power_law(x_data, a, b)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            print(f"Power law fit: y = {a:.4e} * x^({b:.4f}), R² = {r_squared:.4f}")
            return a, b, r_squared
        except Exception as e:
            print(f"Error fitting power law: {e}")
            return None

    def plot_training_curves(
        self,
        show_all_curves: bool = True,
        show_frontier_only: bool = False,
        show_power_law_fit: bool = True,
        flop_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        use_all_points: bool = True,
    ) -> None:
        """
        Plot training curves for all experiments.

        Args:
            show_all_curves: Whether to show all training curves
            show_frontier_only: Whether to only show frontier experiments
            show_power_law_fit: Whether to show power law fit
            flop_range: Optional tuple of (min_flops, max_flops) to recalculate frontier for this plot
            figsize: Figure size
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)

        # If a flop range is specified, recalculate frontier for this plot
        if flop_range is not None:
            # Store original frontier
            original_frontier = self.frontier_points.copy()
            # Calculate new frontier with flop range
            self.identify_frontier(
                method="pareto", flop_range=flop_range, use_all_points=use_all_points
            )

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

        # Plot frontier markers
        if use_all_points:
            # Plot frontier using all intermediate points as stars
            frontier_pts = getattr(self, "frontier_points_all", [])
            for name, comp, loss in frontier_pts:
                color = (
                    self.experiments[name]["color"] if name in self.experiments else "k"
                )
                plt.scatter(
                    comp,
                    loss,
                    color=color,
                    s=160,
                    marker="*",
                    zorder=120,
                    edgecolors="black",
                    linewidth=1.5,
                    label=None,
                )
        else:
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
        has_frontier = (
            use_all_points and len(getattr(self, "frontier_points_all", [])) >= 2
        ) or (not use_all_points and bool(self.frontier_points))
        if show_power_law_fit and has_frontier:
            power_law_result = self.fit_power_law(use_all_points=use_all_points)
            if power_law_result is not None:
                a, b, r_squared = power_law_result
                # Generate fit curve
                if use_all_points:
                    xs = [
                        comp
                        for (_, comp, _) in getattr(self, "frontier_points_all", [])
                    ]
                else:
                    xs = [exp["final_compute"] for exp in experiments_to_plot.values()]
                min_compute = min(xs)
                max_compute = max(xs)
                x_fit = np.logspace(np.log10(min_compute), np.log10(max_compute), 100)
                y_fit = a * np.power(x_fit, b)

                plt.plot(
                    x_fit,
                    y_fit,
                    "k--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Power law fit: y = {a:.2e} * x^({b:.3f}) (R² = {r_squared:.3f})",
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
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=25, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

        # Restore original frontier if we modified it for this plot
        if flop_range is not None:
            self.frontier_points = original_frontier


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

    # Add experiments - you can modify these paths and names as needed
    experiments_config = [
        {
            "name": "transformer 32d mup",
            "csv_path": "../experimental_data_folder/mup_scaling_experiments/mup_32d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 32,
        },
        {
            "name": "transformer 40d mup",
            "csv_path": "../experimental_data_folder/mup_scaling_experiments/mup_40d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 40,
        },
        {
            "name": "transformer 48d mup",
            "csv_path": "../experimental_data_folder/mup_scaling_experiments/mup_48d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 48,
        },
        {
            "name": "transformer 56d mup",
            "csv_path": "../experimental_data_folder/mup_scaling_experiments/mup_56d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 56,
        },
        {
            "name": "transformer 64d mup",
            "csv_path": "../experimental_data_folder/mup_scaling_experiments/mup_64d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 64,
        },
        {
            "name": "optimal lr sgd 32d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_32d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "mup_transformer",
            "hidden_dim": 64,
        },
        {
            "name": "optimal lr sgd 32d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_32d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "optimal_lr_sgd_transformer",
            "hidden_dim": 32,
        },
        {
            "name": "optimal lr sgd 40d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_40d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "optimal_lr_sgd_transformer",
            "hidden_dim": 40,
        },
        {
            "name": "optimal lr sgd 48d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_40d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "optimal_lr_sgd_transformer",
            "hidden_dim": 48,
        },
        {
            "name": "optimal lr sgd 56d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_56d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "optimal_lr_sgd_transformer",
            "hidden_dim": 56,
        },
        {
            "name": "optimal lr sgd 64d",
            "csv_path": "../experimental_data_folder/optimal_lr_sgd_scaling/optimal_lr_sgd_64d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "optimal_lr_sgd_transformer",
            "hidden_dim": 64,
        },


        {
            "name": "vanilla optimal lr",
            "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_32d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "vanilla_transformer",
            "hidden_dim": 32,
        },
        {
            "name": "vanilla optimal lr",
            "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_40d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "vanilla_transformer",
            "hidden_dim": 40,
        },
        {
            "name": "vanilla optimal lr",
            "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_48d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "vanilla_transformer",
            "hidden_dim": 48,
        },
        {
            "name": "vanilla optimal lr",
            "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_56d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "vanilla_transformer",
            "hidden_dim": 56,
        },
        {
            "name": "vanilla optimal lr",
            "csv_path": "../experimental_data_folder/vanilla_scaling_optimal_lr/vanilla_64d.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "vanilla_transformer",
            "hidden_dim": 64,
        },






    ]

    #   {
    #         "name": "lstm 16d optimal",
    #         "csv_path": "../experimental_data_folder/lstm_sgd_scaling/lstm_16d_sgd.csv",
    #         "marker": "o",
    #         "include_in_frontier": True,  # Include in frontier analysis
    #         "class": "sgd_lstm",
    #         "hidden_dim": 16,
    #     },
    #     {
    #         "name": "lstm 24d optimal",
    #         "csv_path": "../experimental_data_folder/lstm_sgd_scaling/lstm_24d_sgd.csv",
    #         "marker": "o",
    #         "include_in_frontier": True,  # Include in frontier analysis
    #         "class": "sgd_lstm",
    #         "hidden_dim": 24,
    #     },
    #     {
    #         "name": "lstm 32d optimal",
    #         "csv_path": "../experimental_data_folder/lstm_sgd_scaling/lstm_32d_sgd.csv",
    #         "marker": "o",
    #         "include_in_frontier": True,  # Include in frontier analysis
    #         "class": "sgd_lstm",
    #         "hidden_dim": 32,
    #     },
    #     {
    #         "name": "lstm 48d optimal",
    #         "csv_path": "../experimental_data_folder/lstm_sgd_scaling/lstm_48d_sgd.csv",
    #         "marker": "o",
    #         "include_in_frontier": True,  # Include in frontier analysis
    #         "class": "sgd_lstm",
    #         "hidden_dim": 48,
    #     },
    #     {
    #         "name": "lstm 64d optimal",
    #         "csv_path": "../experimental_data_folder/lstm_sgd_scaling/lstm_64d_sgd.csv",
    #         "marker": "o",
    #         "include_in_frontier": True,  # Include in frontier analysis
    #         "class": "sgd_lstm",
    #         "hidden_dim": 64,
    #     },

    # Add experiments
    for config in experiments_config:
        analyzer.add_experiment(
            name=config["name"],
            csv_path=config["csv_path"],
            color=config.get("color"),
            marker=config.get("marker", "o"),
            include_in_frontier=config.get("include_in_frontier", True),
            class_name=config.get("class"),
            hidden_dim=config.get("hidden_dim"),
        )

    # Identify per-class frontiers
    analyzer.identify_frontier_by_class(
        method="pareto",
        classes=["mup_transformer", "optimal_lr_sgd_transformer", "vanilla_transformer"],
        flop_range_by_class={
            "mup_transformer": (4 * 1e14, 1e15),
            "optimal_lr_sgd_transformer": (1e14, 1e15),
            "vanilla_transformer": (1e14, 1e15),
        },
    )

    # Example 1: Plot all experiments with frontier analysis
    analyzer.plot_training_curves_by_class(
        show_all_curves=True,
        show_power_law_fit=True,
        save_path="Figures/universal_scaling_law_study_by_class.png",
        classes_to_plot=["mup_transformer", "optimal_lr_sgd_transformer", "vanilla_transformer"],
        flop_range_by_class={
            "mup_transformer": (1e14, 1e15),
            "optimal_lr_sgd_transformer": (1e14, 1e15),
            "vanilla_transformer": (1e14, 1e15),
        },
        colormap="viridis",  # Color experiments by hidden dimension
    )

    # Example 2: Plot with specific FLOP range to focus on a region
    # Uncomment and modify the range as needed:
    # analyzer.plot_training_curves(
    #     show_all_curves=True,
    #     show_frontier_only=False,
    #     show_power_law_fit=True,
    #     flop_range=(1e16, 1e18),  # Example: focus on 10^16 to 10^18 FLOPs
    #     save_path="Figures/universal_scaling_law_study_range.png",
    # )

# %%
