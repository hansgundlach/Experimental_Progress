# %%
"""
Single-Panel Figure: Transformer vs LSTM Scaling with 2-Layer LSTM Highlighted

Same as the left panel of neurips_two_panel_scaling, but standalone.
2-layer LSTM runs are overlaid in red (no fit line for them).
"""

import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
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

TrainingCurveAnalyzer = getattr(nextgen_module, "TrainingCurveAnalyzer")
IRREDUCIBLE_LOSS = 1.9
FONT_CONFIG = getattr(nextgen_module, "FONT_CONFIG")
ALPHA_CONFIG = getattr(nextgen_module, "ALPHA_CONFIG")

NEURIPS_FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 16,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 16,
    "minor_tick_size": 16,
    "legend_size": 13,
    "fit_label_size": 10,
    "equation_size": 17,
    "annotation_size": 15,
    "target_loss_label_size": 14,
}

USE_THEORETICAL_FLOPS = False

class_legend_mapping = {
    "lstm": "LSTM",
    "lstm_sgd": "LSTM SGD",
    "transformer": "Transformer",
    "sgd": "SGD",
    "2017 Transformer": "2017 Transformer",
    "sin transformer": "2017 Transformer",
}

analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping=class_legend_mapping,
)

# Load experiments_config from nextgen module file (same as stupid_ipython.py)
with open(module_path, "r") as f:
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
        raise ValueError(f"Could not find 'experiments_config = [' in {module_path}")

if experiments_config:
    for exp in experiments_config:
        if "include_in_in_frontier" in exp:
            exp["include_in_frontier"] = exp.pop("include_in_in_frontier")
else:
    raise ValueError("Failed to extract experiments_config")

# Same override as stupid_ipython.py
classes_that_need_fits = ["transformer", "lstm", "sin transformer", "sgd", "2017 Transformer"]
for exp in experiments_config:
    if exp.get("class") in classes_that_need_fits:
        if exp.get("include_in_frontier") is False:
            exp["include_in_frontier"] = True

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


# ============================================================================
# SEPARATE ANALYZER for 2-layer LSTM (so it doesn't affect main fits)
# ============================================================================

analyzer_layer2 = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
)

data_base = Path(__file__).parent.parent / "experimental_data_folder" / "x1_lstm_layer1"
layer2_experiments = [
    {"name": "320d LSTM 2-layer", "csv_path": str(data_base / "320d_layer2.csv"), "hidden_dim": 320},
    {"name": "384d LSTM 2-layer", "csv_path": str(data_base / "384d_layer2.csv"), "hidden_dim": 384},
    {"name": "448d LSTM 2-layer", "csv_path": str(data_base / "448d_layer2.csv"), "hidden_dim": 448},
    {"name": "512d LSTM 2-layer", "csv_path": str(data_base / "512d_layer2.csv"), "hidden_dim": 512},
]

for exp in layer2_experiments:
    analyzer_layer2.add_experiment(
        name=exp["name"],
        csv_path=exp["csv_path"],
        color="tab:red",
        marker="s",
        include_in_frontier=True,
        class_name="lstm_layer2",
        hidden_dim=exp["hidden_dim"],
    )


# ============================================================================
# REUSE plot_panel_to_axis from stupid_ipython for the main content
# ============================================================================

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
    analyzer.identify_frontier_by_class(
        method="pareto",
        flop_range=None,
        use_all_points=True,
        classes=classes_to_plot,
        flop_range_by_class=flop_range_by_class,
    )

    plotted_classes = set()

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

    for cls in classes_to_plot:
        pts = analyzer.frontier_points_all_by_class.get(cls, [])
        for exp_name, comp, loss in pts:
            if exp_name in analyzer.experiments:
                color = analyzer.experiments[exp_name].get("color", "tab:blue")
            else:
                color = "k"

            ax.scatter(
                comp, loss,
                color=color,
                s=120,
                marker="*",
                zorder=120,
                edgecolors="black",
                linewidth=1.5,
                alpha=ALPHA_CONFIG["frontier_points_alpha"],
                label=None,
            )

    if show_power_law_fit:
        fit_results = analyzer.fit_power_law_by_class(
            class_names=classes_to_plot, use_all_points=True
        )
        for cls in classes_to_plot:
            params = fit_results.get(cls)
            if params is None:
                continue
            a, b, r2 = params

            xs = [comp for (_, comp, _) in analyzer.frontier_points_all_by_class.get(cls, [])]
            if len(xs) < 2:
                continue

            extended_min, extended_max = extrapolation_range
            x_fit = np.logspace(np.log10(extended_min), np.log10(extended_max), 200)
            y_fit = a * np.power(x_fit, b)

            if cls in analyzer.class_legend_mapping:
                legend_label = analyzer.class_legend_mapping[cls]
            else:
                legend_label = cls

            class_color = analyzer.get_class_color(cls)

            if show_legend:
                ax.plot(x_fit, y_fit, "--", linewidth=3, alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                        label=f"{legend_label} fit:\n{a:.1f} · C^({b:.3f})", color=class_color)
            else:
                ax.plot(x_fit, y_fit, "--", linewidth=3, alpha=ALPHA_CONFIG["power_law_fit_alpha"], color=class_color)

                if equation_positions is not None and cls in equation_positions:
                    eq_x, eq_y = equation_positions[cls]
                else:
                    mid_idx = len(x_fit) // 2
                    eq_x = x_fit[mid_idx]
                    eq_y = y_fit[mid_idx] * 1.3

                equation_text = f"${a:.1f} \\cdot C^{{{b:.3f}}}$"
                ax.text(
                    eq_x, eq_y, equation_text,
                    fontsize=NEURIPS_FONT_CONFIG["equation_size"],
                    color="black", weight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8,
                              edgecolor=class_color, linewidth=2),
                )

    all_target_lines = []
    if target_loss_lines:
        all_target_lines.extend(target_loss_lines)
    if target_loss_line:
        all_target_lines.append(target_loss_line)

    if all_target_lines and show_power_law_fit:
        for line_config in all_target_lines:
            target_loss = line_config.get("loss")
            compare_classes = line_config.get("classes", [])
            line_color = line_config.get("color", "black")
            theoretical_law = line_config.get("theoretical_law")

            if len(compare_classes) == 2:
                fit_results = analyzer.fit_power_law_by_class(class_names=compare_classes, use_all_points=True)
                intersections = {}
                for cls in compare_classes:
                    if theoretical_law is not None and cls == theoretical_law.get("class"):
                        E = theoretical_law.get("E")
                        A = theoretical_law.get("A")
                        gamma = theoretical_law.get("gamma")
                        L0 = analyzer.irreducible_loss
                        excess_loss = target_loss - (E - L0)
                        if gamma != 0 and A > 0 and excess_loss > 0:
                            intersections[cls] = np.power(excess_loss / A, 1.0 / gamma)
                    elif cls in fit_results and fit_results[cls] is not None:
                        a, b, _ = fit_results[cls]
                        if b != 0 and a > 0 and target_loss > 0:
                            intersections[cls] = np.power(target_loss / a, 1.0 / b)

                if len(intersections) == 2:
                    class1, class2 = compare_classes
                    compute1 = intersections.get(class1)
                    compute2 = intersections.get(class2)

                    if compute1 is not None and compute2 is not None:
                        if compute1 > compute2:
                            compute1, compute2 = compute2, compute1
                            class1, class2 = class2, class1

                        compute_ratio = float(compute2) / float(compute1)

                        ax.axhline(y=target_loss, color="black", linestyle="--",
                                   linewidth=2.5, alpha=0.8, zorder=85,
                                   label=f"Target Loss: {target_loss:.2f}")

                        ax.text(0.02, target_loss * 1.05, f"{target_loss:.1f}",
                                transform=ax.get_yaxis_transform(),
                                fontsize=NEURIPS_FONT_CONFIG["target_loss_label_size"],
                                color="black", ha="left", va="bottom", zorder=95)

                        ax.plot([float(compute1), float(compute2)], [target_loss, target_loss],
                                color=line_color, linestyle="-", linewidth=5, alpha=1.0, zorder=90)

                        mid_compute = np.sqrt(float(compute1) * float(compute2))
                        annotation_y_multiplier = line_config.get("annotation_y_multiplier", 1.3)
                        y_position = float(target_loss) * annotation_y_multiplier

                        ax.annotate(
                            f"{compute_ratio:.1f}x",
                            xy=(mid_compute, target_loss),
                            xytext=(mid_compute, y_position),
                            fontsize=NEURIPS_FONT_CONFIG["annotation_size"],
                            color="black", weight="bold", ha="center", va="bottom",
                            arrowprops=dict(arrowstyle="->", color="black", lw=2, alpha=0.7),
                            zorder=100,
                        )
                        print(f"Target loss {target_loss:.2f}: {class1}={compute1:.2e}, {class2}={compute2:.2e}, ratio={compute_ratio:.2f}x")

    ax.set_xlabel("Compute (FLOPs)", fontsize=NEURIPS_FONT_CONFIG["xlabel_size"])
    if show_ylabel:
        ax.set_ylabel("Validation Loss - Irreducible ", fontsize=NEURIPS_FONT_CONFIG["ylabel_size"])
    ax.set_yscale("log")
    ax.set_xscale("log")

    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=NEURIPS_FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=NEURIPS_FONT_CONFIG["minor_tick_size"])

    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=NEURIPS_FONT_CONFIG["title_size"], fontweight="bold", va="top", ha="left")

    if title:
        ax.set_title(title, fontsize=NEURIPS_FONT_CONFIG["title_size"],
                     fontweight=NEURIPS_FONT_CONFIG["title_weight"])

    if show_legend:
        leg = ax.legend(loc="upper right", fontsize=NEURIPS_FONT_CONFIG["legend_size"],
                        framealpha=0.9, prop={"weight": "bold"})
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None)
        if handles:
            for h in handles:
                if hasattr(h, "set_alpha"):
                    h.set_alpha(1.0)


# ============================================================================
# CREATE SINGLE-PANEL FIGURE
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

target_lines = [
    {
        "loss": (5.277 - 1.9),
        "classes": ["transformer", "lstm"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.3,
    },
    {
        "loss": 2.5,
        "classes": ["transformer", "lstm"],
        "color": "orange",
        "linestyle": "-",
        "linewidth": 4,
        "alpha": 1.0,
        "annotation_y_multiplier": 1.2,
    },
]

plot_panel_to_axis(
    analyzer=analyzer,
    ax=ax,
    classes_to_plot=["transformer", "lstm"],
    flop_range_by_class={
        "transformer": (1e16, 5e17),
        "lstm": (1e16, 5e17),
    },
    extrapolation_range=(10**14.0, 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=None,
    panel_label=None,
    target_loss_lines=target_lines,
    equation_positions={
        "transformer": (2e14, 2.0),
        "lstm": (5e16, 5.7),
    },
    show_ylabel=True,
    title="Modern Transformer vs LSTM Scaling",
)

# ============================================================================
# OVERLAY 2-layer LSTM in red (data only, no fit)
# ============================================================================

layer2_plotted = False
for name, exp in analyzer_layer2.experiments.items():
    compute_vals = exp["data"][exp["compute_col"]].values
    loss_vals = exp["data"][exp["loss_col"]].values - analyzer_layer2.irreducible_loss
    mask = loss_vals > 0

    if np.any(mask):
        label = "LSTM (2 layers)" if not layer2_plotted else None
        layer2_plotted = True
        ax.plot(
            compute_vals[mask],
            loss_vals[mask],
            marker="s",
            linestyle=exp["linestyle"],
            color="tab:red",
            alpha=ALPHA_CONFIG["data_points_alpha"],
            label=label,
            linewidth=1.5,
            markersize=3,
        )

# Add legend now that all classes are plotted
leg = ax.legend(
    loc="upper right",
    fontsize=NEURIPS_FONT_CONFIG["legend_size"],
    framealpha=0.9,
    prop={"weight": "bold"},
)
handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None)
if handles:
    for h in handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1.0)

plt.tight_layout()

save_path = "Figures/single_panel_lstm_layers.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {save_path}")

save_path_pdf = "Figures/single_panel_lstm_layers.pdf"
plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
print(f"Figure saved to {save_path_pdf}")

plt.show()

# %%
