# %%
"""
Training Curve Analysis with Compute Column Toggle and Multiple Fit Options

This script provides a TrainingCurveAnalyzer class that can analyze training curves
and fit multiple types of scaling laws. You can easily toggle between using
'theoretical_flops' and 'total_flops_profiler' as the compute column by setting the
use_theoretical_flops parameter when initializing the analyzer.
The analyzer supports two types of fits:
1. Power law fit: y = a * x^b
2. Sklearn-style fit: L = E + A * C^alpha

The analyzer also supports:
- Class-to-legend-label mapping for cleaner legends
- Flexible color specification using colormaps (e.g., 'viridis[2]', 'plasma[0.3]')

Usage:
    # To use theoretical_flops:
    analyzer = TrainingCurveAnalyzer(use_theoretical_flops=True)

    # To use total_flops_profiler (default):
    analyzer = TrainingCurveAnalyzer(use_theoretical_flops=False)

    # With class legend mapping for cleaner legends:
    class_mapping = {"lstm": "LSTM experiments", "transformer": "Transformer experiments"}
    analyzer = TrainingCurveAnalyzer(class_legend_mapping=class_mapping)

    # Add experiment with colormap color:
    analyzer.add_experiment("exp1", "data.csv", color="viridis[2]")

    # To enable sklearn-style fitting:
    analyzer.plot_training_curves_by_class(show_sklearn_fit=True)
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional
import warnings
import re

# sklearn imports removed - not actually used in the code

# =============================================================================
# FONT SIZE CONFIGURATION
# =============================================================================
# Modify these values to adjust font sizes throughout the plots

FONT_CONFIG = {
    # Main plot elements
    "xlabel_size": 25,
    "ylabel_size": 25,
    "title_size": 25,
    "title_weight": "bold",
    # Tick labels
    "major_tick_size": 20,
    "minor_tick_size": 20,
    # Legend
    "legend_size": 20,
    # Power law fit labels
    "fit_label_size": 12,
    # Theoretical scaling law labels
    "theory_label_size": 12,
    # Regular plot elements (for plot_training_curves method)
    "regular_xlabel_size": 12,
    "regular_ylabel_size": 12,
    "regular_title_size": 14,
    "regular_legend_size": 15,
}

# =============================================================================
# ALPHA (TRANSPARENCY) CONFIGURATION
# =============================================================================
# Modify these values to adjust transparency of plot elements

ALPHA_CONFIG = {
    # Data points and lines
    "data_points_alpha": 0.15,  # Alpha for training curve data points
    "data_lines_alpha": 0.8,  # Alpha for training curve lines
    # Frontier points (stars)
    "frontier_points_alpha": 0.00,  # Alpha for frontier star points
    # Fit lines
    "power_law_fit_alpha": 0.8,  # Alpha for power law fit lines
    "sklearn_fit_alpha": 0.8,  # Alpha for sklearn fit lines
    "theoretical_alpha": 0.8,  # Alpha for theoretical scaling law lines
}

IRREDUCIBLE_LOSS = 1.8

# =============================================================================
# CLASS COLOR CONFIGURATION
# =============================================================================
# Centralized color configuration for each class - modify these to change colors
# All experiments of the same class will use the same color
# Available colormaps: viridis, plasma, inferno, magma, cividis, tab10, etc.
# Format: "colormap[index]" where index is 0.0-1.0 for continuous colormaps
CLASS_COLORS = {
    "lstm": "viridis[0.8]",  # Green-ish color
    "transformer": "viridis[0.0]",  # Dark purple color
    "sgd": "viridis[0.6]",  # Yellow-green color
    "2017 Transformer": "plasma[0.7]",  # Orange color
    "lstm_sgd": "tab:orange",  # Orange color for LSTM SGD experiments
    "default": "tab:blue",  # Default color for unclassified experiments
}


class TrainingCurveAnalyzer:
    """
    A class to analyze and plot training curves from multiple CSV files,
    identifying frontier points and fitting power laws.
    """

    def __init__(
        self,
        irreducible_loss: float = IRREDUCIBLE_LOSS,
        use_theoretical_flops: bool = False,
        class_legend_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            irreducible_loss: The irreducible loss to subtract from validation losses
            use_theoretical_flops: If True, use 'theoretical_flops' column; if False, use 'total_flops_profiler'
            class_legend_mapping: Optional mapping from class names to legend labels (e.g., {'lstm': 'LSTM experiments'})
        """
        self.irreducible_loss = irreducible_loss
        self.use_theoretical_flops = use_theoretical_flops
        self.default_compute_col = (
            "theoretical_flops" if use_theoretical_flops else "total_flops_profiler"
        )
        self.experiments = {}
        self.frontier_points = []
        self.color_palette = list(mcolors.TABLEAU_COLORS.values())
        # Per-class frontier storage
        self.frontier_points_by_class: Dict[str, List[str]] = {}
        self.frontier_points_all_by_class: Dict[str, List[Tuple[str, float, float]]] = (
            {}
        )
        # Class to legend label mapping
        self.class_legend_mapping = class_legend_mapping or {}

    def _parse_color_spec(self, color_spec: str):
        """
        Parse color specification that can be either a standard color name or a colormap index.

        Args:
            color_spec: Color specification like 'red', 'tab:blue', 'viridis[2]', 'plasma[0.5]'

        Returns:
            Actual color value that matplotlib can use
        """
        if not isinstance(color_spec, str):
            return color_spec

        # Check if it's a colormap specification like 'viridis[2]' or 'plasma[0.5]'
        colormap_match = re.match(r"(\w+)\[([^\]]+)\]", color_spec)
        if colormap_match:
            cmap_name = colormap_match.group(1)
            index_str = colormap_match.group(2)

            try:
                # Get the colormap (updated for matplotlib 3.7+)
                try:
                    cmap = cm.get_cmap(cmap_name)
                except AttributeError:
                    # Fallback for newer matplotlib versions
                    import matplotlib.pyplot as plt

                    cmap = plt.colormaps.get_cmap(cmap_name)

                # Parse the index (can be integer or float)
                if "." in index_str:
                    index = float(index_str)
                else:
                    index = int(index_str)

                # For continuous colormaps, ensure index is in [0, 1] range
                if cmap_name in ["viridis", "plasma", "inferno", "magma", "cividis"]:
                    # These are continuous colormaps that expect values in [0, 1]
                    if index > 1.0:
                        index = 1.0
                    elif index < 0.0:
                        index = 0.0
                else:
                    # For discrete colormaps, normalize to [0, 1] if needed
                    if index > 1.0:
                        index = 1.0
                    elif index < 0.0:
                        index = 0.0

                # Get color from colormap
                color = cmap(index)
                print(f"Colormap '{color_spec}' -> {color}")  # Debug print
                return color

            except (ValueError, KeyError) as e:
                print(
                    f"Warning: Could not parse colormap '{color_spec}': {e}. Using as-is."
                )
                return color_spec

        # If not a colormap specification, return as-is
        return color_spec

    def add_experiment(
        self,
        name: str,
        csv_path: str,
        compute_col: Optional[str] = None,
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
            compute_col: Column name for compute values (uses default if None)
            loss_col: Column name for loss values
            color: Color for plotting (auto-assigned if None). Can be:
                   - Standard color name: 'red', 'blue', 'tab:orange'
                   - Colormap index: 'viridis[2]', 'plasma[0.3]', 'inferno[5]'
            marker: Marker style for plotting
            linestyle: Line style for plotting
            alpha: Transparency for plotting
            include_in_frontier: Whether to include this experiment in frontier analysis
            class_name: Class/category name for the experiment
            hidden_dim: Hidden dimension size for the model
        """
        try:
            df = pd.read_csv(csv_path)

            # Use default compute column if not specified
            if compute_col is None:
                compute_col = self.default_compute_col

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
                # Use centralized class color if available, otherwise use palette
                if class_name and class_name in CLASS_COLORS:
                    color = CLASS_COLORS[class_name]
                else:
                    color = self.color_palette[
                        len(self.experiments) % len(self.color_palette)
                    ]

            # Always parse color specification (handles colormap indices like 'viridis[2]')
            color = self._parse_color_spec(color)

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

    def get_class_color(self, class_name: str):
        """
        Get the color for a given class by finding the first experiment of that class.

        Args:
            class_name: Name of the class

        Returns:
            The color used by experiments of this class, or 'black' if no experiments found
        """
        for exp in self.experiments.values():
            if exp.get("class", "default") == class_name:
                return exp["color"]
        return "black"  # fallback color

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
                # Use better initial guesses and bounds for power law fitting
                # Initial guess: a = mean(y) / mean(x)^(-0.1), b = -0.1 (typical scaling exponent)
                initial_guess = [
                    np.mean(y_data) / np.power(np.mean(x_data), -0.1),
                    -0.1,
                ]
                # Bounds: a in [1e-10, 1e10], b in [-1, 0.5]
                bounds = ([1e-10, -1], [1e10, 0.5])

                params, _ = curve_fit(
                    power_law,
                    x_data,
                    y_data,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000,
                )
                a, b = params
                y_pred = power_law(x_data, a, b)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(
                    f"Class '{cls}' power law fit: {a:.4e} * x^({b:.4f}), R² = {r_squared:.4f}"
                )
                results[cls] = (a, b, r_squared)
            except Exception as e:
                print(f"Error fitting power law for class '{cls}': {e}")
                results[cls] = None

        return results

    def fit_sklearn_curve_by_class(
        self,
        class_names: Optional[List[str]] = None,
        use_all_points: bool = True,
    ) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
        """
        Fit L = E + AC^(alpha) separately for each class using sklearn.

        Args:
            class_names: Which classes to fit. Defaults to all present in stored frontiers.
            use_all_points: Whether to fit using all-point frontier or final-point frontier.

        Returns:
            Mapping from class name to (E, A, alpha, r_squared), or None if fit fails/insufficient points.
        """
        # Determine classes present
        if class_names is None:
            if use_all_points:
                class_names = list(self.frontier_points_all_by_class.keys())
            else:
                class_names = list(self.frontier_points_by_class.keys())

        results: Dict[str, Optional[Tuple[float, float, float, float]]] = {}

        def sklearn_curve(x, E, A, alpha):
            """L = E + A * C^(alpha)"""
            return E + A * np.power(x, alpha)

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

            if x_data.size < 3:  # Need at least 3 points for 3 parameters
                print(f"Need at least 3 points to fit sklearn curve for class '{cls}'")
                results[cls] = None
                continue

            try:
                # Use scipy curve_fit for the sklearn-style formula with better initial guesses
                # Initial guess: E = min(y), A = 1, alpha = -0.1 (typical scaling exponent)
                initial_guess = [np.min(y_data), 1.0, -0.1]
                # Bounds: E in [0, max(y)], A in [1e-10, 1e10], alpha in [-1, 0.5]
                bounds = ([0, 1e-10, -1], [np.max(y_data), 1e10, 0.5])

                params, _ = curve_fit(
                    sklearn_curve,
                    x_data,
                    y_data,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000,
                )
                E, A, alpha = params
                y_pred = sklearn_curve(x_data, E, A, alpha)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(
                    f"Class '{cls}' sklearn curve fit: L = {E:.4f} + {A:.4e} * C^({alpha:.4f}), R² = {r_squared:.4f}"
                )
                results[cls] = (E, A, alpha, r_squared)
            except Exception as e:
                print(f"Error fitting sklearn curve for class '{cls}': {e}")
                results[cls] = None

        return results

    def fit_sklearn_curve(
        self,
        experiment_names: Optional[List[str]] = None,
        use_all_points: bool = True,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Fit L = E + AC^(alpha) to the specified experiments using sklearn-style fitting.

        Args:
            experiment_names: List of experiment names to fit (uses frontier if None)
            use_all_points: Whether to fit using all-point frontier or final-point frontier

        Returns:
            Tuple of (E, A, alpha, r_squared) parameters for L = E + AC^(alpha), or None if fitting fails
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
                print("No experiments to fit sklearn curve to")
                return None
            for name in experiment_names:
                if name in self.experiments:
                    exp = self.experiments[name]
                    if exp["include_in_frontier"]:
                        x_data.append(exp["final_compute"])
                        y_data.append(exp["final_loss"])

        if len(x_data) < 3:
            print("Need at least 3 points to fit sklearn curve")
            return None

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Define the sklearn-style curve: L = E + A * C^(alpha)
        def sklearn_curve(x, E, A, alpha):
            return E + A * np.power(x, alpha)

        try:
            # Use better initial guesses and bounds for convergence
            # Initial guess: E = min(y), A = 1, alpha = -0.1 (typical scaling exponent)
            initial_guess = [np.min(y_data), 1.0, -0.1]
            # Bounds: E in [0, max(y)], A in [1e-10, 1e10], alpha in [-1, 0.5]
            bounds = ([0, 1e-10, -1], [np.max(y_data), 1e10, 0.5])

            params, _ = curve_fit(
                sklearn_curve,
                x_data,
                y_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
            E, A, alpha = params

            # Calculate R-squared
            y_pred = sklearn_curve(x_data, E, A, alpha)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            print(
                f"Sklearn curve fit: L = {E:.4f} + {A:.4e} * C^({alpha:.4f}), R² = {r_squared:.4f}"
            )
            return E, A, alpha, r_squared
        except Exception as e:
            print(f"Error fitting sklearn curve: {e}")
            return None

    def plot_theoretical_scaling_law(
        self,
        E: float,
        A: float,
        gamma: float,
        compute_range: Optional[Tuple[float, float]] = None,
        label: str = "Theoretical Scaling Law",
        color: str = "red",
        linestyle: str = "-",
        linewidth: float = 3,
        alpha: float = 0.8,
    ) -> None:
        """
        Plot a theoretical scaling law of the form E + A*C^(gamma) with irreducible loss subtracted.

        Args:
            E: The irreducible loss parameter (will be subtracted by self.irreducible_loss)
            A: The scaling coefficient
            gamma: The scaling exponent
            compute_range: Optional tuple of (min_compute, max_compute) for plotting range
            label: Label for the plot legend
            color: Color for the line
            linestyle: Line style
            linewidth: Line width
            alpha: Transparency
        """
        # If no compute range specified, use the range of all experiments
        if compute_range is None:
            all_computes = []
            for exp in self.experiments.values():
                all_computes.extend(exp["data"][exp["compute_col"]].values)
            if not all_computes:
                print("No experiments loaded to determine compute range")
                return
            min_compute = min(all_computes)
            max_compute = max(all_computes)
            # Extend range slightly for better visualization
            compute_range = (min_compute * 0.1, max_compute * 10)

        min_compute, max_compute = compute_range

        # Generate compute values for plotting
        compute_vals = np.logspace(np.log10(min_compute), np.log10(max_compute), 200)

        # Calculate theoretical loss: E - L0 + A*C^(gamma)
        # where L0 is the irreducible loss
        theoretical_loss = (E - self.irreducible_loss) + A * np.power(
            compute_vals, gamma
        )

        # Plot the theoretical curve
        plt.plot(
            compute_vals,
            theoretical_loss,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f"{label}: L = {E:.3f} - {self.irreducible_loss:.3f} + {A:.2e} * C^({gamma:.3f})",
        )

    def plot_training_curves_by_class(
        self,
        show_all_curves: bool = True,
        show_power_law_fit: bool = True,
        show_sklearn_fit: bool = False,
        flop_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        use_all_points: bool = True,
        classes_to_plot: Optional[List[str]] = None,
        flop_range_by_class: Optional[Dict[str, Tuple[float, float]]] = None,
        extrapolation_factor: float = 2.0,
        theoretical_scaling_laws: Optional[List[Dict]] = None,
    ) -> None:
        """
        Plot training curves and per-class frontiers and fits.
        Shows experiment names in legend with colors determined by the color parameter.

        Args:
            show_all_curves: Whether to show all training curves
            show_power_law_fit: Whether to show power law fit (y = a * x^b)
            show_sklearn_fit: Whether to show sklearn-style fit (L = E + A * C^alpha)
            flop_range: Optional tuple of (min_flops, max_flops) to recalculate frontier for this plot
            figsize: Figure size
            save_path: Path to save the plot
            use_all_points: Whether to use all training points or just final points
            classes_to_plot: Which classes to plot
            flop_range_by_class: Per-class FLOP ranges
            extrapolation_factor: Factor to extend trend lines beyond data range (e.g., 2.0 = 2x range)
            theoretical_scaling_laws: Optional list of dicts with keys 'E', 'A', 'gamma', 'label', 'color', 'linestyle'
                                    to plot theoretical scaling laws of form E + A*C^(gamma)
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

        # Filter classes to plot
        if classes_to_plot is None:
            classes_to_plot = sorted(
                {exp.get("class", "default") for exp in self.experiments.values()}
            )

        # Optionally plot all training curves
        if show_all_curves:
            # Track which classes we've already plotted to avoid duplicate legend entries
            plotted_classes = set()

            for name, exp in self.experiments.items():
                cls = exp.get("class", "default")
                if cls not in classes_to_plot:
                    continue

                # Use the color parameter directly
                color = exp.get("color", "tab:blue")

                compute_vals = exp["data"][exp["compute_col"]].values
                loss_vals = exp["data"][exp["loss_col"]].values - self.irreducible_loss
                mask = loss_vals > 0
                if np.any(mask):
                    # Use class legend mapping if available, otherwise use class name
                    if cls in self.class_legend_mapping:
                        legend_label = self.class_legend_mapping[cls]
                    else:
                        legend_label = cls

                    # Only add label if we haven't plotted this class yet
                    label = legend_label if cls not in plotted_classes else None
                    plotted_classes.add(cls)

                    ax.plot(
                        compute_vals[mask],
                        loss_vals[mask],
                        marker=exp["marker"],
                        linestyle=exp["linestyle"],
                        color=color,
                        alpha=ALPHA_CONFIG["data_points_alpha"],
                        label=label,
                        linewidth=2,
                        markersize=6,
                    )

        # Plot class-specific frontier points as stars
        if use_all_points:
            for cls in classes_to_plot:
                pts = self.frontier_points_all_by_class.get(cls, [])
                for name, comp, loss in pts:
                    if name in self.experiments:
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
                        alpha=ALPHA_CONFIG["frontier_points_alpha"],
                        label=None,
                    )
        else:
            for cls in classes_to_plot:
                names = self.frontier_points_by_class.get(cls, [])
                for name in names:
                    exp = self.experiments[name]
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
                        alpha=ALPHA_CONFIG["frontier_points_alpha"],
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

                # Extend the range using extrapolation_factor
                compute_range = max_compute - min_compute
                extended_min = min_compute / extrapolation_factor
                extended_max = max_compute * extrapolation_factor

                x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
                y_fit = a * np.power(x_fit, b)

                # Use a unique linestyle per class for clarity; color matches class
                linestyle = "--"
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle,
                    linewidth=4,
                    alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                    label=f"{cls} power law: \n {a:.2e} * x^({b:.3f})",
                    color=self.get_class_color(cls),
                )

        # label=f"{cls} power law: \n y = {a:.2e} * x^({b:.3f}) (R² = {r2:.3f})",
        # Plot per-class sklearn-style fits
        if show_sklearn_fit:
            sklearn_fit_results = self.fit_sklearn_curve_by_class(
                class_names=classes_to_plot, use_all_points=use_all_points
            )
            for cls, params in sklearn_fit_results.items():
                if params is None:
                    continue
                E, A, alpha, r2 = params
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

                # Extend the range using extrapolation_factor
                extended_min = min_compute / extrapolation_factor
                extended_max = max_compute * extrapolation_factor

                x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
                y_fit = E + A * np.power(x_fit, alpha)
                # Use a different linestyle for sklearn fit; color matches class
                linestyle = "-."
                ax.plot(
                    x_fit,
                    y_fit,
                    linestyle,
                    linewidth=4,
                    alpha=ALPHA_CONFIG["sklearn_fit_alpha"],
                    label=f"{cls} sklearn: \n L = {E:.3f} + {A:.2e} * C^({alpha:.3f}) (R² = {r2:.3f})",
                    color=self.get_class_color(cls),
                )

        # Plot theoretical scaling laws if provided
        if theoretical_scaling_laws:
            for law_config in theoretical_scaling_laws:
                E = law_config.get("E", IRREDUCIBLE_LOSS)  # Default to irreducible loss
                A = law_config.get("A", 1.0)
                gamma = law_config.get("gamma", -0.1)
                label = law_config.get("label", "Theoretical Scaling Law")
                color = law_config.get("color", "red")
                linestyle = law_config.get("linestyle", "-")
                linewidth = law_config.get("linewidth", 3)
                alpha = law_config.get("alpha", 0.8)

                # Determine compute range for theoretical curve
                if use_all_points:
                    all_xs = []
                    for cls in classes_to_plot:
                        pts = self.frontier_points_all_by_class.get(cls, [])
                        all_xs.extend([comp for (_, comp, _) in pts])
                else:
                    all_xs = []
                    for cls in classes_to_plot:
                        names = self.frontier_points_by_class.get(cls, [])
                        all_xs.extend(
                            [self.experiments[n]["final_compute"] for n in names]
                        )

                if all_xs:
                    min_compute = min(all_xs)
                    max_compute = max(all_xs)
                    # Extend range using extrapolation_factor
                    extended_min = min_compute / extrapolation_factor
                    extended_max = max_compute * extrapolation_factor

                    x_theory = np.logspace(
                        np.log10(extended_min), np.log10(extended_max), 200
                    )
                    # Calculate theoretical loss: E - L0 + A*C^(gamma)
                    y_theory = (E - self.irreducible_loss) + A * np.power(
                        x_theory, gamma
                    )
                    # actual E with irreducible loss removed
                    irred_removed_E = E - self.irreducible_loss
                    ax.plot(
                        x_theory,
                        y_theory,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=ALPHA_CONFIG["theoretical_alpha"],
                        label=f"{label}: \nL = {irred_removed_E:.3f} + {A:.2e} * C^({gamma:.3f})",
                    )

            # L = {E:.3f} - {self.irreducible_loss:.3f} + {A:.2e} * C^({gamma:.3f})",

        ax.set_xlabel("Compute (FLOPS)", fontsize=FONT_CONFIG["xlabel_size"])
        ax.set_ylabel(
            "Validation Loss - Irreducible", fontsize=FONT_CONFIG["ylabel_size"]
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "Transformer Scaling Analysis",
            fontsize=FONT_CONFIG["title_size"],
            fontweight=FONT_CONFIG["title_weight"],
        )
        # Increase tick label sizes
        ax.tick_params(
            axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"]
        )
        ax.tick_params(
            axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"]
        )

        # Position legend
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=FONT_CONFIG["legend_size"],
        )

        # Adjust layout to accommodate legend
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
            # Use better initial guesses and bounds for power law fitting
            # Initial guess: a = mean(y) / mean(x)^(-0.1), b = -0.1 (typical scaling exponent)
            initial_guess = [np.mean(y_data) / np.power(np.mean(x_data), -0.1), -0.1]
            # Bounds: a in [1e-10, 1e10], b in [-1, 0.5]
            bounds = ([1e-10, -1], [1e10, 0.5])

            params, covariance = curve_fit(
                power_law, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=10000
            )
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
        show_sklearn_fit: bool = False,
        flop_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        use_all_points: bool = True,
        extrapolation_factor: float = 2.0,
        theoretical_scaling_laws: Optional[List[Dict]] = None,
    ) -> None:
        """
        Plot training curves for all experiments.

        Args:
            show_all_curves: Whether to show all training curves
            show_frontier_only: Whether to only show frontier experiments
            show_power_law_fit: Whether to show power law fit (y = a * x^b)
            show_sklearn_fit: Whether to show sklearn-style fit (L = E + A * C^alpha)
            flop_range: Optional tuple of (min_flops, max_flops) to recalculate frontier for this plot
            figsize: Figure size
            save_path: Path to save the plot
            use_all_points: Whether to use all training points or just final points
            extrapolation_factor: Factor to extend trend lines beyond data range (e.g., 2.0 = 2x range)
            theoretical_scaling_laws: Optional list of dicts with keys 'E', 'A', 'gamma', 'label', 'color', 'linestyle'
                                    to plot theoretical scaling laws of form E + A*C^(gamma)
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
                    alpha=ALPHA_CONFIG["data_lines_alpha"],
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
                    alpha=ALPHA_CONFIG["frontier_points_alpha"],
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
                    alpha=(
                        ALPHA_CONFIG["frontier_points_alpha"]
                        if is_frontier
                        else ALPHA_CONFIG["data_points_alpha"]
                    ),
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

                # Extend the range using extrapolation_factor
                extended_min = min_compute / extrapolation_factor
                extended_max = max_compute * extrapolation_factor

                x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
                y_fit = a * np.power(x_fit, b)

                plt.plot(
                    x_fit,
                    y_fit,
                    "k--",
                    linewidth=4,
                    alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                    label=f"Power law fit: y = {a:.2e} * x^({b:.3f}) (R² = {r_squared:.3f})",
                )

        # Plot sklearn-style fit if requested
        if show_sklearn_fit and has_frontier:
            sklearn_result = self.fit_sklearn_curve(use_all_points=use_all_points)
            if sklearn_result is not None:
                E, A, alpha, r_squared = sklearn_result
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

                # Extend the range using extrapolation_factor
                extended_min = min_compute / extrapolation_factor
                extended_max = max_compute * extrapolation_factor

                x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
                y_fit = E + A * np.power(x_fit, alpha)

                plt.plot(
                    x_fit,
                    y_fit,
                    "r-.",
                    linewidth=4,
                    alpha=ALPHA_CONFIG["sklearn_fit_alpha"],
                    label=f"Sklearn fit: L = {E:.3f} + {A:.2e} * C^({alpha:.3f}) (R² = {r_squared:.3f})",
                )

        # Plot theoretical scaling laws if provided
        if theoretical_scaling_laws:
            for law_config in theoretical_scaling_laws:
                E = law_config.get("E", 1.9)  # Default to irreducible loss
                A = law_config.get("A", 1.0)
                gamma = law_config.get("gamma", -0.1)
                label = law_config.get("label", "Theoretical Scaling Law")
                color = law_config.get("color", "red")
                linestyle = law_config.get("linestyle", "-")
                linewidth = law_config.get("linewidth", 3)
                alpha = law_config.get("alpha", 0.8)

                # Determine compute range for theoretical curve
                if use_all_points:
                    xs = [
                        comp
                        for (_, comp, _) in getattr(self, "frontier_points_all", [])
                    ]
                else:
                    xs = [exp["final_compute"] for exp in experiments_to_plot.values()]

                if xs:
                    min_compute = min(xs)
                    max_compute = max(xs)
                    # Extend range using extrapolation_factor
                    extended_min = min_compute / extrapolation_factor
                    extended_max = max_compute * extrapolation_factor

                    x_theory = np.logspace(
                        np.log10(extended_min), np.log10(extended_max), 200
                    )
                    # Calculate theoretical loss: E - L0 + A*C^(gamma)
                    y_theory = (E - self.irreducible_loss) + A * np.power(
                        x_theory, gamma
                    )

                    plt.plot(
                        x_theory,
                        y_theory,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=ALPHA_CONFIG["theoretical_alpha"],
                        label=f"{label}: L = {E:.3f} - {self.irreducible_loss:.3f} + {A:.2e} * C^({gamma:.3f})",
                    )

        # Customize plot
        plt.xlabel("Compute (FLOPS)", fontsize=FONT_CONFIG["regular_xlabel_size"])
        plt.ylabel(
            "Validation Loss - Irreducible", fontsize=FONT_CONFIG["regular_ylabel_size"]
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.title(
            "Training Curves and Scaling Analysis",
            fontsize=FONT_CONFIG["regular_title_size"],
            fontweight="bold",
        )

        # Add legend
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=FONT_CONFIG["regular_legend_size"],
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
    # Toggle between theoretical_flops and total_flops_profiler by setting use_theoretical_flops

    # Add experiments - you can modify these paths and names as needed
    experiments_config = [
        {
            "name": "32d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/32_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 32,
        },
        # 64 96 128 160 corrected melis scaling
        {
            "name": "64d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/64_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 64,
        },
        {
            "name": "96d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/96_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 96,
        },
        {
            "name": "128d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/128_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 128,
        },
        {
            "name": "160d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/160_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 160,
        },
        {
            "name": "192d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/192_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 192,
        },
        {
            "name": "256d corrected melis scaling experiments",
            "csv_path": "../experimental_data_folder/lstm_scaling_study/256_correction_bs64.csv",
            "marker": "o",
            "include_in_in_frontier": False,  # Include in frontier analysis
            "class": "lstm",
            "hidden_dim": 256,
        },
        # # lstm sgd
        # {
        #     "name": "48d melis sgd",
        #     "csv_path": "../experimental_data_folder/lstm_sgd/48melis_steam_sgd.csv",
        #     "marker": "o",
        #     "include_in_frontier": False,  # Include in frontier analysis
        #     "class": "lstm_sgd",
        #     "hidden_dim": 48,
        # },
        # # 64melis scaling
        # {
        #     "name": "64d melis sgd",
        #     "csv_path": "../experimental_data_folder/lstm_sgd/64melis_steam_sgd.csv",
        #     "marker": "o",
        #     "include_in_frontier": False,  # Include in frontier analysis
        #     "class": "lstm_sgd",
        #     "hidden_dim": 64,
        # },
        # # 128 melis scaling
        # {
        #     "name": "128d melis sgd",
        #     "csv_path": "../experimental_data_folder/lstm_sgd/128melis_stream_sgd.csv",
        #     "marker": "o",
        #     "include_in_frontier": False,  # Include in frontier analysis
        #     "class": "lstm_sgd",
        #     "hidden_dim": 128,
        # },
        # transformer scaling further
        # results with swiglu
        # {
        #     "name": "32d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_32d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 32,
        # },
        # {
        #     "name": "48d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_48d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 48,
        # },
        # {
        #     "name": "64d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_64d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 64,
        # },
        # # 96 128 160
        # {
        #     "name": "96d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/96d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 96,
        # },
        # {
        #     "name": "128d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_128d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 128,
        # },
        # {
        #     "name": "160d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_160d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 160,
        # },
        # # 256
        # {
        #     "name": "256d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/swiglu_256d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 256,
        # },
        {
            "name": "32d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/32d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 32,
        },
        {
            "name": "48d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/48d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 48,
        },
        {
            "name": "64d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/64d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 64,
        },
        # # 96 128 160
        # {
        #     "name": "96d transformer scaling further",
        #     "csv_path": "../experimental_data_folder/transformer_scaling/96d_transformer_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "transformer",
        #     "hidden_dim": 96,
        # },
        {
            "name": "128d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/128d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 128,
        },
        {
            "name": "160d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/160d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 160,
        },
        # 256
        {
            "name": "256d transformer scaling further",
            "csv_path": "../experimental_data_folder/transformer_scaling/256d_transformer_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "transformer",
            "hidden_dim": 256,
        },
        # sgd scaling further
        {
            "name": "orig 32d sgd scaling further",
            "csv_path": "../experimental_data_folder/sgd_scaling/32d_sgdbs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "sgd",
            "hidden_dim": 32,
        },
        # {
        #     "name": "orig 48d sgd scaling further",
        #     "csv_path": "../experimental_data_folder/sgd_scaling/48d_sgdbs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "sgd",
        #     # Example: using viridis colormap index 2
        #     "hidden_dim": 48,
        # },
        {
            "name": "orig 64d sgd scaling further",
            "csv_path": "../experimental_data_folder/sgd_scaling/64d_sgdbs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "sgd",
            # Example: using plasma colormap at 0.3
            "hidden_dim": 64,
        },
        # {
        #     "name": "orig 96d sgd scaling further",
        #     "csv_path": "../experimental_data_folder/sgd_scaling/96d_sgdbs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "sgd",
        #     "hidden_dim": 96,
        # },
        {
            "name": "orig 128d sgd scaling further",
            "csv_path": "../experimental_data_folder/sgd_scaling/128d_sgdbs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "sgd",
            "hidden_dim": 128,
        },
        {
            "name": "orig 160d sgd scaling further",
            "csv_path": "../experimental_data_folder/sgd_scaling/160d_sgdbs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "sgd",
            "hidden_dim": 160,
        },
        # look at scaling of 2017 transfomere
        {
            "name": "32d 2017 transformer",
            "csv_path": "../experimental_data_folder/historical_experiments/p32transformer_2017_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "2017 Transformer",
            "hidden_dim": 32,
        },
        # {
        #     "name": "48d 2017 transformer",
        #     "csv_path": "../experimental_data_folder/historical_experiments/p48transformer_2017_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "2017 Transformer",
        #     "hidden_dim": 48,
        # },
        {
            "name": "64d 2017 transformer",
            "csv_path": "../experimental_data_folder/historical_experiments/p64transformer_2017_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "2017 Transformer",
            "hidden_dim": 64,
        },
        # 80 96 128
        # {
        #     "name": "80d 2017 transformer",
        #     "csv_path": "../experimental_data_folder/historical_experiments/p80transformer_2017_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "2017 Transformer",
        #     "hidden_dim": 80,
        # },
        # {
        #     "name": "96d 2017 transformer",
        #     "csv_path": "../experimental_data_folder/historical_experiments/p96transformer_2017_bs64.csv",
        #     "marker": "o",
        #     "include_in_frontier": True,  # Include in frontier analysis
        #     "class": "2017 Transformer",
        #     "hidden_dim": 96,
        # },
        {
            "name": "128d 2017 transformer",
            "csv_path": "../experimental_data_folder/historical_experiments/p128transformer_2017_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "2017 Transformer",
            "hidden_dim": 128,
        },
        {
            "name": "160d 2017 transformer",
            "csv_path": "../experimental_data_folder/historical_experiments/p160transformer_2017_bs64.csv",
            "marker": "o",
            "include_in_frontier": True,  # Include in frontier analysis
            "class": "2017 Transformer",
            "hidden_dim": 160,
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

    USE_THEORETICAL_FLOPS = (
        False  # Set to True to use theoretical_flops, False for total_flops_profiler
    )

    # Define class-to-legend-label mapping for cleaner legend
    class_legend_mapping = {
        "lstm": "LSTM experiments",
        "lstm_sgd": "LSTM SGD experiments",
        "transformer": "Transformer experiments",
        "sgd": "SGD experiments",
        "2017 Transformer": "2017 Transformer experiments",
    }

    analyzer = TrainingCurveAnalyzer(
        irreducible_loss=IRREDUCIBLE_LOSS,
        use_theoretical_flops=USE_THEORETICAL_FLOPS,
        class_legend_mapping=class_legend_mapping,
    )

    # Add experiments
    for config in experiments_config:
        analyzer.add_experiment(
            name=config["name"],
            csv_path=config["csv_path"],
            compute_col=config.get("compute_col"),  # Will use default if not specified
            color=config.get("color"),
            marker=config.get("marker", "o"),
            include_in_frontier=config.get("include_in_frontier", True),
            class_name=config.get("class"),
            hidden_dim=config.get("hidden_dim"),
        )

    # Identify per-class frontiers
    # analyzer.identify_frontier_by_class(
    #     method="pareto",
    #     classes=[
    #         # "optimal_lr_sgd_transformer",
    #         "vanilla_transformer",
    #         "optimal_lr_sgd_transformer",
    #         "vanilla_transformer_no_rotary",
    #         "vanilla_transformer_rmsprop",
    #         "optimal_lr_sgd_transformer",
    #     ],
    #     flop_range_by_class={
    #         # "optimal_lr_sgd_transformer": (1e14, 1e15),
    #         "vanilla_transformer": (1e14, 1e15),
    #         "optimal_lr_sgd_transformer": (1e14, 1e15),
    #         "vanilla_transformer_no_rotary": (1e14, 1e15),
    #         "vanilla_transformer_rmsprop": (1e14, 1e15),
    #     },
    # )

    # Example 1: Plot all experiments with frontier analysis
    analyzer.plot_training_curves_by_class(
        show_all_curves=True,
        show_power_law_fit=True,
        show_sklearn_fit=False,  # Enable sklearn-style fit: L = E + A * C^alpha
        save_path="Figures/universal_scaling_law_study_by_class.png",
        classes_to_plot=[
            # "optimal_lr_sgd_transformer",
            "transformer",
            "lstm",
            # "2017 Transformer",
            # "sgd",
            # "vanilla_transformer_rmsprop",
        ],
        flop_range_by_class={
            "transformer": (1e14, 5 * 1e17),
            # "lstm": (1e14, 1e16 * 5),
            # "2017 Transformer": (1e14, 1e17),
            "sgd": (1e14, 1e17),
        },
        extrapolation_factor=10.0,  # Extend trend lines 3x beyond data range
        # Example theoretical scaling laws to compare with data
        theoretical_scaling_laws=[
            {
                "E": 1.68,  # Irreducible loss
                "A": 87.16,  # Scaling coefficient (larger to be visible)
                "gamma": -0.092,  # Scaling exponent
                "label": "Porian et al. (2025)",
                "color": "red",
                "linestyle": "--",
                "linewidth": 3,
                "alpha": 0.8,
            },
        ],
    )

# %%
