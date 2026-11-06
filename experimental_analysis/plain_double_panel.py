# %%
"""
NeurIPS-Style Two-Panel Figure Generator

Creates a publication-quality figure with two side-by-side plots:
- Left: Transformer vs LSTM scaling comparison
- Right: Sin transformer with theoretical scaling law comparison

Optimized for NeurIPS two-column format.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from graphing_utils import TrainingCurveAnalyzer, FONT_CONFIG, ALPHA_CONFIG

IRREDUCIBLE_LOSS = 1.9
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"

# NeurIPS-optimized font configuration
NEURIPS_FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 14,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 12,
    "minor_tick_size": 10,
    "legend_size": 11,  # Increased for bold text readability
    "fit_label_size": 10,
}

# Configuration
USE_THEORETICAL_FLOPS = False

# Define class-to-legend-label mapping for cleaner legend
class_legend_mapping = {
    "lstm_layer1": "LSTM (1 layer, x1)",
    "lstm_layer2": "LSTM (2 layers, x2)",
    "historical_transformer": "Historical Transformer",
    "modern_transformer": "Modern Transformer",
}

# Initialize analyzer
analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping=class_legend_mapping,
)

def add_dim_sweep(configs, *, class_name, label, folder, dims, filename_for_dim, color):
    for dim in dims:
        csv_path = DATA_ROOT / folder / filename_for_dim(dim)
        if not csv_path.exists():
            print(f"Skipping missing CSV: {csv_path}")
            continue
        configs.append(
            {
                "name": f"{dim}d {label}",
                "csv_path": str(csv_path),
                "marker": "o",
                "include_in_frontier": True,
                "class": class_name,
                "hidden_dim": dim,
                "color": color,
            }
        )


experiments_config = []
add_dim_sweep(
    experiments_config,
    class_name="lstm_layer1",
    label="LSTM layer 1",
    folder="x1_lstm_layer1",
    dims=[32, 48, 64, 128, 160, 192, 224, 256, 320, 384, 448, 512],
    filename_for_dim=lambda dim: f"{dim}d.csv",
    color="viridis[0.80]",
)
add_dim_sweep(
    experiments_config,
    class_name="lstm_layer1",
    label="LSTM layer 1",
    folder="lstm_layer1",
    dims=[80, 96, 112],
    filename_for_dim=lambda dim: f"{dim}d.csv",
    color="viridis[0.80]",
)
add_dim_sweep(
    experiments_config,
    class_name="lstm_layer2",
    label="LSTM layer 2",
    folder="x2_lstm_layer2",
    dims=[32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 448, 512],
    filename_for_dim=lambda dim: f"{dim}d.csv",
    color="cividis[0.65]",
)
add_dim_sweep(
    experiments_config,
    class_name="historical_transformer",
    label="Historical Transformer",
    folder="x1_historical",
    dims=[32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320],
    filename_for_dim=lambda dim: f"{dim}_all_reset.csv",
    color="plasma[0.70]",
)
add_dim_sweep(
    experiments_config,
    class_name="modern_transformer",
    label="Modern Transformer",
    folder="x2_modern",
    dims=[32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256],
    filename_for_dim=lambda dim: (
        f"{dim}_modern.csv" if dim in (48, 112) else f"{dim}_modern_40.csv"
    ),
    color="viridis[0.15]",
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
    "lstm_layer1",
    "lstm_layer2",
    "historical_transformer",
    "modern_transformer",
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
    fit_summaries = []
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

            fit_summaries.append(
                f"{legend_label}: A={a:.2e}, alpha={b:.3f}, R^2={r2:.3f}"
            )
            ax.plot(
                x_fit,
                y_fit,
                "--",
                linewidth=3,
                alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                label=f"{legend_label} fit:\n{a:.2e} * C^({b:.3f}), R^2={r2:.3f}",
                color=analyzer.get_class_color(cls),
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
            show_constant = law_config.get(
                "show_constant", True
            )  # Option to hide E term

            extended_min, extended_max = extrapolation_range
            x_theory = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
            y_theory = (E - analyzer.irreducible_loss) + A * np.power(x_theory, gamma)
            irred_removed_E = E - analyzer.irreducible_loss

            # Build label based on whether to show constant term
            if show_constant:
                equation_label = f"{irred_removed_E:.3f} + {A:.2e} × C^({gamma:.3f})"
            else:
                equation_label = f"{A:.2e} × C^({gamma:.3f})"

            ax.plot(
                x_theory,
                y_theory,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=ALPHA_CONFIG["theoretical_alpha"],
                label=f"{label}:\n{equation_label}",
            )

    # Formatting
    ax.set_xlabel("Compute (FLOPs)", fontsize=NEURIPS_FONT_CONFIG["xlabel_size"])
    ax.set_ylabel(
        "Validation Loss - Irreducible", fontsize=NEURIPS_FONT_CONFIG["ylabel_size"]
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(
        axis="both", which="major", labelsize=NEURIPS_FONT_CONFIG["major_tick_size"]
    )
    ax.tick_params(
        axis="both", which="minor", labelsize=NEURIPS_FONT_CONFIG["minor_tick_size"]
    )

    if fit_summaries:
        ax.text(
            0.03,
            0.04,
            "\n".join(fit_summaries),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.88,
            },
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

    # Legend
    leg = ax.legend(
        loc="upper right",
        framealpha=0.9,
        prop={"weight": "bold", "size": NEURIPS_FONT_CONFIG["legend_size"]},
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
# %%
# NeurIPS two-column width is typically ~7 inches
fig = plt.figure(figsize=(14, 5.5))

# Create grid for two panels
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

# Panel 1: LSTM layer 1/layer 2 vs historical transformer
ax1 = fig.add_subplot(gs[0, 0])
plot_panel_to_axis(
    analyzer=analyzer,
    ax=ax1,
    classes_to_plot=["lstm_layer1", "lstm_layer2", "historical_transformer"],
    flop_range_by_class={
        "lstm_layer1": (1e14, 5 * 1e17),
        "lstm_layer2": (1e14, 5 * 1e17),
        "historical_transformer": (1e14, 5 * 1e17),
    },
    extrapolation_range=(10 ** (14.0), 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=None,
    panel_label="(a)",
)

# Panel 2: modern transformer
ax2 = fig.add_subplot(gs[0, 1])
plot_panel_to_axis(
    analyzer=analyzer,
    ax=ax2,
    classes_to_plot=["modern_transformer"],
    flop_range_by_class={
        "modern_transformer": (1e14, 5 * 1e17),
    },
    extrapolation_range=(10 ** (14.0), 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=None,
    panel_label="(b)",
)

# Overall title (optional - comment out if not needed for paper)
fig.suptitle(
    "Scaling Analysis: LSTMs, Historical Transformer, and Modern Transformer",
    fontsize=NEURIPS_FONT_CONFIG["title_size"] + 2,
    fontweight=NEURIPS_FONT_CONFIG["title_weight"],
    y=0.98,
)

plt.tight_layout()

# Save the figure
save_path = ROOT / "Figures" / "two_panel_lstm_historical_modern_scaling.png"
save_path.parent.mkdir(exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"NeurIPS two-panel figure saved to {save_path.resolve()}")

# Also save as PDF for publication
save_path_pdf = ROOT / "Figures" / "two_panel_lstm_historical_modern_scaling.pdf"
plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
print(f"NeurIPS two-panel figure saved to {save_path_pdf.resolve()}")

plt.show()

# %%
