# %%
"""
NeurIPS-Style Two-Panel Figure Generator

Creates a publication-quality figure with two side-by-side plots:
- Left: Transformer vs LSTM scaling comparison
- Right: Sin transformer with theoretical scaling law comparison

Optimized for NeurIPS two-column format.
"""

import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker

# Import TrainingCurveAnalyzer from the file with space in name
module_path = Path(__file__).parent / "nextgen_lstmvtransformer copy.py"
spec = importlib.util.spec_from_file_location(
    "nextgen_lstmvtransformer_copy", module_path
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")
nextgen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nextgen_module)

# Import the class and constants we need
TrainingCurveAnalyzer = getattr(nextgen_module, "TrainingCurveAnalyzer")
# IRREDUCIBLE_LOSS = getattr(nextgen_module, "IRREDUCIBLE_LOSS")
IRREDUCIBLE_LOSS = 1.9
FONT_CONFIG = getattr(nextgen_module, "FONT_CONFIG")
ALPHA_CONFIG = getattr(nextgen_module, "ALPHA_CONFIG")

# NeurIPS-optimized font configuration
NEURIPS_FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 16,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 16,
    "minor_tick_size": 16,
    "legend_size": 13,  # Increased for bold text readability
    "fit_label_size": 10,
    "equation_size": 17,  # Font size for scaling law equations
    "annotation_size": 15,  # Font size for compute ratio annotations (e.g., "10.5x")
    "target_loss_label_size": 14,  # Font size for target loss value labels (e.g., "3.4", "2.5")
}

# Configuration
USE_THEORETICAL_FLOPS = False

# Define class-to-legend-label mapping for cleaner legend
class_legend_mapping = {
    "lstm": "LSTM",
    "lstm_sgd": "LSTM SGD",
    "transformer": "Transformer",
    "sgd": "SGD",
    "2017 Transformer": "2017 Transformer",
    "sin transformer": "2017 Transformer",
}

# Initialize analyzer
analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping=class_legend_mapping,
)

# Load experiments_config from the same file we loaded the TrainingCurveAnalyzer from
experiments_config_module_path = module_path  # Use the same path we already defined

with open(experiments_config_module_path, "r") as f:
    lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "experiments_config = [" in line:
            start_idx = i
            break

    if start_idx is not None:
        bracket_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(lines)):
            bracket_count += lines[i].count("[")
            bracket_count -= lines[i].count("]")
            if bracket_count == 0 and i > start_idx:
                end_idx = i
                break

        config_lines = lines[start_idx : end_idx + 1]
        min_indent = float("inf")
        for line in config_lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        if min_indent != float("inf"):
            config_lines = [
                line[min_indent:] if line.strip() else line for line in config_lines
            ]

        config_code = "".join(config_lines)
        namespace = {"__builtins__": __builtins__}
        exec(config_code, namespace)
        experiments_config = namespace.get("experiments_config")
    else:
        raise ValueError(
            f"Could not find 'experiments_config = [' in {experiments_config_module_path}"
        )

# Fix any typos in the config
if experiments_config:
    for exp in experiments_config:
        if "include_in_in_frontier" in exp:
            exp["include_in_frontier"] = exp.pop("include_in_in_frontier")
else:
    raise ValueError("Failed to extract experiments_config")

# Override include_in_frontier for classes we need
classes_that_need_fits = [
    "transformer",
    "lstm",
    "sin transformer",
    "sgd",
    "2017 Transformer",
]

if experiments_config:
    for exp in experiments_config:
        exp_class = exp.get("class")
        if exp_class in classes_that_need_fits:
            if exp.get("include_in_frontier") is False:
                exp["include_in_frontier"] = True

# Add experiments
for config in experiments_config:
    analyzer.add_experiment(
        name=config["name"],
        csv_path=config["csv_path"],
        compute_col=config.get("compute_col"),
        color=config.get("color"),
        marker=config.get("marker", "o"),
        include_in_frontier=config.get("include_in_frontier", True),
        class_name=config.get("class"),
        hidden_dim=config.get("hidden_dim"),
    )


def plot_panel_to_axis(
    analyzer,
    ax,
    classes_to_plot,
    flop_range_by_class,
    extrapolation_range,
    show_power_law_fit=True,
    theoretical_scaling_laws=None,
    panel_label=None,
    target_loss_line=None,
    target_loss_lines=None,
    show_legend=False,
    equation_positions=None,
    show_ylabel=True,
    title=None,
):
    """
    Plot training curves to a specific matplotlib axis for multi-panel figures.

    Args:
        analyzer: TrainingCurveAnalyzer instance
        ax: matplotlib axis to plot on
        classes_to_plot: List of class names to include
        flop_range_by_class: Dict mapping class names to (min_flops, max_flops)
        extrapolation_range: Tuple of (min_compute, max_compute) for fit extrapolation
        show_power_law_fit: Whether to show power law fits
        theoretical_scaling_laws: Optional list of theoretical scaling law dicts
        panel_label: Optional label like "(a)" or "(b)" for the panel
        target_loss_line: Optional dict for a single target loss line (legacy)
        target_loss_lines: Optional list of dicts for multiple target loss lines.
                          Each dict can include 'annotation_y_multiplier' (default: 1.3) to control
                          the vertical position of the compute ratio annotation. Higher values move it up.
        show_legend: Whether to show the legend (default: True)
        equation_positions: Optional dict mapping class names to (x, y) positions for equation annotations.
                          Position is in data coordinates. Example: {'lstm': (1e15, 2.0), 'transformer': (1e16, 1.5)}
                          If None, equations will be positioned automatically.
        show_ylabel: Whether to show the y-axis label (default: True)
        title: Optional title for the subplot
    """
    # Compute frontiers for these classes
    analyzer.identify_frontier_by_class(
        method="pareto",
        flop_range=None,
        use_all_points=True,
        classes=classes_to_plot,
        flop_range_by_class=flop_range_by_class,
    )

    # Track which classes we've plotted for legend
    plotted_classes = set()

    # Plot all training curves
    for name, exp in analyzer.experiments.items():
        cls = exp.get("class", "default")
        if cls not in classes_to_plot:
            continue

        color = exp.get("color", "tab:blue")
        compute_vals = exp["data"][exp["compute_col"]].values
        loss_vals = exp["data"][exp["loss_col"]].values - analyzer.irreducible_loss
        mask = loss_vals > 0

        if np.any(mask):
            if cls in analyzer.class_legend_mapping:
                legend_label = analyzer.class_legend_mapping[cls]
            else:
                legend_label = cls

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
                linewidth=1.5,
                markersize=3,
            )

    # Plot frontier points as stars
    for cls in classes_to_plot:
        pts = analyzer.frontier_points_all_by_class.get(cls, [])
        for exp_name, comp, loss in pts:
            if exp_name in analyzer.experiments:
                color = analyzer.experiments[exp_name].get("color", "tab:blue")
            else:
                color = "k"

            ax.scatter(
                comp,
                loss,
                color=color,
                s=120,
                marker="*",
                zorder=120,
                edgecolors="black",
                linewidth=1.5,
                alpha=ALPHA_CONFIG["frontier_points_alpha"],
                label=None,
            )

    # Plot power law fits
    if show_power_law_fit:
        fit_results = analyzer.fit_power_law_by_class(
            class_names=classes_to_plot, use_all_points=True
        )
        for cls in classes_to_plot:
            params = fit_results.get(cls)
            if params is None:
                continue
            a, b, r2 = params

            xs = [
                comp
                for (_, comp, _) in analyzer.frontier_points_all_by_class.get(cls, [])
            ]
            if len(xs) < 2:
                continue

            extended_min, extended_max = extrapolation_range
            x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
            y_fit = a * np.power(x_fit, b)

            # Use legend mapping if available
            if cls in analyzer.class_legend_mapping:
                legend_label = analyzer.class_legend_mapping[cls]
            else:
                legend_label = cls

            class_color = analyzer.get_class_color(cls)

            # Plot the fit line (with or without label depending on show_legend)
            if show_legend:
                ax.plot(
                    x_fit,
                    y_fit,
                    "--",
                    linewidth=3,
                    alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                    label=f"{legend_label} fit:\n{a:.1f} · C^({b:.3f})",
                    color=class_color,
                )
            else:
                ax.plot(
                    x_fit,
                    y_fit,
                    "--",
                    linewidth=3,
                    alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                    color=class_color,
                )

                # Add equation annotation next to the fit line
                # Determine position for the equation
                if equation_positions is not None and cls in equation_positions:
                    # Use user-specified position
                    eq_x, eq_y = equation_positions[cls]
                else:
                    # Auto-position: place at middle of fit line
                    mid_idx = len(x_fit) // 2
                    eq_x = x_fit[mid_idx]
                    eq_y = y_fit[mid_idx] * 1.3  # Slightly above the line

                # Format equation with proper exponents using LaTeX
                equation_text = f"${a:.1f} \\cdot C^{{{b:.3f}}}$"

                ax.text(
                    eq_x,
                    eq_y,
                    equation_text,
                    fontsize=NEURIPS_FONT_CONFIG["equation_size"],
                    color="black",
                    weight="bold",
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=class_color,
                        linewidth=2,
                    ),
                )

    # Plot theoretical scaling laws
    if theoretical_scaling_laws:
        for law_config in theoretical_scaling_laws:
            E = law_config.get("E", IRREDUCIBLE_LOSS)
            A = law_config.get("A", 1.0)
            gamma = law_config.get("gamma", -0.1)
            label = law_config.get("label", "Theoretical")
            color = law_config.get("color", "red")
            linestyle = law_config.get("linestyle", "-")
            linewidth = law_config.get("linewidth", 3)
            alpha_val = law_config.get("alpha", 0.8)
            show_irreducible_loss = law_config.get("show_irreducible_loss", True)

            extended_min, extended_max = extrapolation_range
            x_theory = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
            y_theory = (E - analyzer.irreducible_loss) + A * np.power(x_theory, gamma)
            irred_removed_E = E - analyzer.irreducible_loss

            # Format label based on show_irreducible_loss parameter
            if show_irreducible_loss:
                legend_label = (
                    f"{label}:\n{irred_removed_E:.3f} + {A:.1f} · C^({gamma:.3f})"
                )
            else:
                legend_label = f"{label}:\n{A:.1f} · C^({gamma:.3f})"

            ax.plot(
                x_theory,
                y_theory,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=ALPHA_CONFIG["theoretical_alpha"],
                label=legend_label,
            )

    # Plot target loss line(s) and compute ratio if specified
    all_target_lines = []
    if target_loss_lines:
        all_target_lines.extend(target_loss_lines)
    if target_loss_line:
        all_target_lines.append(target_loss_line)

    if all_target_lines and show_power_law_fit:
        for line_config in all_target_lines:
            target_loss = line_config.get("loss")
            compare_classes = line_config.get("classes", [])
            line_style = line_config.get("linestyle", "-")
            line_width = line_config.get("linewidth", 3)
            line_color = line_config.get("color", "black")
            line_alpha = line_config.get("alpha", 1.0)
            theoretical_law = line_config.get(
                "theoretical_law"
            )  # Optional theoretical law parameters

            if len(compare_classes) == 2:
                # Get power law fits for the specified classes
                fit_results = analyzer.fit_power_law_by_class(
                    class_names=compare_classes, use_all_points=True
                )

                # Calculate intersection points: target_loss = a * C^b => C = (target_loss / a)^(1/b)
                intersections = {}
                for cls in compare_classes:
                    # Check if this class should use theoretical law instead
                    if theoretical_law is not None and cls == theoretical_law.get(
                        "class"
                    ):
                        # Use theoretical scaling law: L = (E - L0) + A * C^gamma
                        # Solve for C: target_loss = (E - L0) + A * C^gamma
                        # => C = ((target_loss - (E - L0)) / A)^(1/gamma)
                        E = theoretical_law.get("E")
                        A = theoretical_law.get("A")
                        gamma = theoretical_law.get("gamma")
                        L0 = analyzer.irreducible_loss

                        excess_loss = target_loss - (E - L0)
                        if gamma != 0 and A > 0 and excess_loss > 0:
                            compute_at_target = np.power(excess_loss / A, 1.0 / gamma)
                            intersections[cls] = compute_at_target
                    elif cls in fit_results and fit_results[cls] is not None:
                        a, b, _ = fit_results[cls]
                        if b != 0 and a > 0 and target_loss > 0:
                            compute_at_target = np.power(target_loss / a, 1.0 / b)
                            intersections[cls] = compute_at_target

                if len(intersections) == 2:
                    class1, class2 = compare_classes
                    compute1 = intersections.get(class1)
                    compute2 = intersections.get(class2)

                    if (
                        compute1 is not None
                        and compute2 is not None
                        and isinstance(compute1, (int, float))
                        and isinstance(compute2, (int, float))
                    ):
                        # Ensure compute1 is the smaller value (left) and compute2 is larger (right)
                        if compute1 > compute2:
                            compute1, compute2 = compute2, compute1
                            class1, class2 = class2, class1

                        # Compute ratio
                        compute_ratio = float(compute2) / float(compute1)

                        # Draw dashed horizontal line across entire plot at target loss
                        ax.axhline(
                            y=target_loss,
                            color="black",
                            linestyle="--",
                            linewidth=2.5,
                            alpha=0.8,
                            zorder=85,
                            label=f"Target Loss: {target_loss:.2f}",
                        )

                        # Add text label showing the target loss value on the left side
                        ax.text(
                            0.02,  # 2% from the left edge in axes coordinates
                            target_loss * 1.05,  # Slightly above the line
                            f"{target_loss:.1f}",
                            transform=ax.get_yaxis_transform(),  # Use y-data coords, x-axes coords
                            fontsize=NEURIPS_FONT_CONFIG["target_loss_label_size"],
                            color="black",
                            ha="left",
                            va="bottom",
                            zorder=95,
                        )

                        # Draw solid line segment between the two intersection points
                        ax.plot(
                            [float(compute1), float(compute2)],
                            [target_loss, target_loss],
                            color=line_color,
                            linestyle="-",
                            linewidth=5,
                            alpha=1.0,
                            zorder=90,
                        )

                        # Add annotation with compute ratio in the middle of the shaded region
                        mid_compute = np.sqrt(
                            float(compute1) * float(compute2)
                        )  # Geometric mean for log scale

                        # Position annotation - use custom offset if provided, otherwise default multiplier
                        annotation_y_multiplier = line_config.get(
                            "annotation_y_multiplier", 1.3
                        )
                        y_position = float(target_loss) * annotation_y_multiplier

                        ax.annotate(
                            f"{compute_ratio:.1f}x",
                            xy=(mid_compute, target_loss),
                            xytext=(mid_compute, y_position),
                            fontsize=NEURIPS_FONT_CONFIG["annotation_size"],
                            color="black",
                            weight="bold",
                            ha="center",
                            va="bottom",
                            arrowprops=dict(
                                arrowstyle="->", color="black", lw=2, alpha=0.7
                            ),
                            zorder=100,
                        )

                        print(f"Target loss {target_loss:.2f} achieved at:")
                        print(f"  {class1}: {compute1:.2e} FLOPs")
                        print(f"  {class2}: {compute2:.2e} FLOPs")
                        print(f"  Compute ratio: {compute_ratio:.2f}x")

    # Formatting
    ax.set_xlabel("Compute (FLOPs)", fontsize=NEURIPS_FONT_CONFIG["xlabel_size"])
    if show_ylabel:
        ax.set_ylabel(
            "Validation Loss - Irreducible (Log Scale)",
            fontsize=NEURIPS_FONT_CONFIG["ylabel_size"],
        )
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Set regular notation for Y axis (no scientific notation)
    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)

    ax.grid(True, alpha=0.3)
    ax.tick_params(
        axis="both", which="major", labelsize=NEURIPS_FONT_CONFIG["major_tick_size"]
    )
    ax.tick_params(
        axis="both", which="minor", labelsize=NEURIPS_FONT_CONFIG["minor_tick_size"]
    )

    # Add panel label if provided
    if panel_label:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=NEURIPS_FONT_CONFIG["title_size"],
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Add title if provided
    if title:
        ax.set_title(
            title,
            fontsize=NEURIPS_FONT_CONFIG["title_size"],
            fontweight=NEURIPS_FONT_CONFIG["title_weight"],
        )

    # Legend (only if show_legend is True)
    if show_legend:
        leg = ax.legend(
            loc="upper right",
            fontsize=NEURIPS_FONT_CONFIG["legend_size"],
            framealpha=0.9,
            prop={"weight": "bold"},
        )

        # Force opaque legend markers/lines
        # Handle both old and new matplotlib API
        handles = getattr(leg, "legend_handles", None) or getattr(
            leg, "legendHandles", None
        )
        if handles:
            for h in handles:
                if hasattr(h, "set_alpha"):
                    h.set_alpha(1.0)
                if hasattr(h, "get_facecolors"):
                    fc = h.get_facecolors()
                    if len(fc):
                        fc[:, -1] = 1.0
                        h.set_facecolors(fc)
                if hasattr(h, "get_edgecolors"):
                    ec = h.get_edgecolors()
                    if len(ec):
                        ec[:, -1] = 1.0
                        h.set_edgecolors(ec)


# ============================================================================
# CREATE TWO-PANEL NEURIPS FIGURE
# ============================================================================

# NeurIPS two-column width is typically ~7 inches
fig = plt.figure(figsize=(14, 5.5))

# Create grid for two panels
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

# Panel 1: Transformer vs LSTM
ax1 = fig.add_subplot(gs[0, 0])

# Define shared target loss lines
target_lines = [
    {
        "loss": (5.277 - 1.9),
        "classes": ["transformer", "lstm"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.3,  # Adjust this to move annotation up/down
    },
    {
        "loss": 2.5,
        "classes": ["transformer", "lstm"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.2,  # Adjust this to move annotation up/down
    },
]

plot_panel_to_axis(
    analyzer=analyzer,
    ax=ax1,
    classes_to_plot=["transformer", "lstm"],
    flop_range_by_class={
        "transformer": (1e16, 5 * 1e17),
        "lstm": (1e16, 1e17 * 5),
    },
    extrapolation_range=(10 ** (14.0), 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=None,
    panel_label="(a)",
    target_loss_lines=target_lines,
    equation_positions={
        "transformer": (2e14, 2.0),
        "lstm": (5e16, 5.7),
    },
    show_ylabel=True,
    title="Transformer vs LSTM Scaling",
)

# Panel 2: Sin transformer with theoretical comparison
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # Share Y axis with ax1

# Target lines for panel 2
target_lines_2 = [
    {
        "loss": (5.277 - 1.9),
        "classes": ["sin transformer", "transformer"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.2,  # Adjust this to move annotation up/down
        "theoretical_law": {
            "class": "transformer",
            "E": 1.9,
            "A": 88.0,
            "gamma": -0.094,
        },
    },
    {
        "loss": 2.5,
        "classes": ["sin transformer", "transformer"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.2,  # Adjust this to move annotation up/down
        "theoretical_law": {
            "class": "transformer",
            "E": 1.9,
            "A": 88.0,
            "gamma": -0.094,
        },
    },
]

plot_panel_to_axis(
    analyzer=analyzer,
    ax=ax2,
    classes_to_plot=["sin transformer"],
    flop_range_by_class={
        "sin transformer": (1e16, 1e17 * 5),
    },
    extrapolation_range=(10 ** (14.0), 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=[
        {
            "E": 1.9,
            "A": 85.5,
            "gamma": -0.093,
            "label": "Modern Transformer",
            "color": "purple",
            "linestyle": "--",
            "linewidth": 3,
            "alpha": 0.8,
            "show_constant": False,  # Set to False to hide E term, True to show it
        },
    ],
    panel_label="(b)",
    target_loss_lines=target_lines_2,
    equation_positions={
        "sin transformer": (7e16, 5.7),
    },
    show_ylabel=False,  # Don't repeat ylabel
    title="2017 Transformer vs Modern Scaling",
)

# Remove y-tick labels from the second subplot since they share axis
plt.setp(ax2.get_yticklabels(), visible=False)

plt.tight_layout()

# Save the figure
save_path = "Figures/neurips_two_panel_scaling.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"NeurIPS two-panel figure saved to {save_path}")

# Also save as PDF for publication
save_path_pdf = "Figures/neurips_two_panel_scaling.pdf"
plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
print(f"NeurIPS two-panel figure saved to {save_path_pdf}")

plt.show()

# %%
