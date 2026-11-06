"""Panel-style scaling comparison for x1 historical vs x1 RMS historical."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"
FIGURES_DIR = ROOT / "Figures"
SUMMARY_PATH = Path(__file__).resolve().parent / "x1_historical_vs_rms_summary.csv"

IRREDUCIBLE_LOSS = 1.9
FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 14,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 16,
    "minor_tick_size": 16,
}
ALPHA_CONFIG = {
    "data_points_alpha": 0.15,
    "power_law_fit_alpha": 0.8,
}

DIMS = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
CLASSES = ["x1_historical", "x1_rms_historical"]

CLASS_LABELS = {
    "x1_historical": "Historical Transformer",
    "x1_rms_historical": "Historical Transformer + RMSNorm",
}

CLASS_COLORS = {
    "x1_historical": "plasma[0.70]",
    "x1_rms_historical": "viridis[0.20]",
}

DEFAULT_SAVE_PATH = "Figures/x1_historical_vs_rms_scaling.png"


def parse_color(color_spec):
    if "[" not in color_spec or not color_spec.endswith("]"):
        return color_spec
    cmap_name, raw_index = color_spec[:-1].split("[", 1)
    return plt.get_cmap(cmap_name)(float(raw_index))


def build_experiments_config():
    configs = []
    for cls in CLASSES:
        for dim in DIMS:
            csv_path = DATA_ROOT / cls / f"{dim}_all_reset.csv"
            configs.append(
                {
                    "name": f"{dim}d {CLASS_LABELS[cls]}",
                    "csv_path": str(csv_path),
                    "marker": "o",
                    "include_in_frontier": True,
                    "class": cls,
                    "hidden_dim": dim,
                    "color": CLASS_COLORS[cls],
                }
            )
    return configs


def load_experiments(configs):
    experiments = {}
    for config in configs:
        df = pd.read_csv(config["csv_path"])
        experiments[config["name"]] = {
            "data": df,
            "compute_col": config.get("compute_col", "total_flops_profiler"),
            "loss_col": "validation_loss",
            "color": parse_color(config["color"]),
            "marker": config.get("marker", "o"),
            "linestyle": "-",
            "class": config["class"],
            "hidden_dim": config["hidden_dim"],
        }
    return experiments


def get_frontier_points(
    experiments,
    cls,
    min_compute=None,
    max_compute=None,
):
    points = []
    for name, exp in experiments.items():
        if exp["class"] != cls:
            continue
        compute_vals = exp["data"][exp["compute_col"]].to_numpy(dtype=float)
        loss_vals = (
            exp["data"][exp["loss_col"]].to_numpy(dtype=float) - IRREDUCIBLE_LOSS
        )
        mask = loss_vals > 0
        if min_compute is not None:
            mask &= compute_vals >= min_compute
        if max_compute is not None:
            mask &= compute_vals <= max_compute
        points.extend(
            (name, float(compute), float(loss))
            for compute, loss in zip(compute_vals[mask], loss_vals[mask])
        )

    frontier = []
    best_loss = float("inf")
    for name, compute, loss in sorted(points, key=lambda item: item[1]):
        if loss < best_loss:
            frontier.append((name, compute, loss))
            best_loss = loss
    return frontier


def fit_power_law(frontier):
    x_data = np.array([point[1] for point in frontier], dtype=float)
    y_data = np.array([point[2] for point in frontier], dtype=float)
    slope, intercept = np.polyfit(np.log(x_data), np.log(y_data), 1)
    coefficient = float(np.exp(intercept))
    exponent = float(slope)
    y_pred = coefficient * np.power(x_data, exponent)
    ss_res = float(np.sum((y_data - y_pred) ** 2))
    ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    return coefficient, exponent, r2


def fit_and_write_summary(experiments, classes_to_plot, flop_range_by_class=None):
    frontiers = {}
    fit_results = {}
    for cls in classes_to_plot:
        min_compute, max_compute = (None, None)
        if flop_range_by_class and cls in flop_range_by_class:
            min_compute, max_compute = flop_range_by_class[cls]
        frontiers[cls] = get_frontier_points(experiments, cls, min_compute, max_compute)
        fit_results[cls] = fit_power_law(frontiers[cls]) if len(frontiers[cls]) >= 2 else None

    rows = []
    for cls in classes_to_plot:
        params = fit_results.get(cls)
        if params is None:
            continue
        coefficient, exponent, r2 = params
        rows.append(
            {
                "class": cls,
                "label": CLASS_LABELS[cls],
                "coefficient": coefficient,
                "scaling_exponent": exponent,
                "r_squared": r2,
                "frontier_points": len(frontiers[cls]),
                "law": f"{coefficient:.6e} * C^({exponent:.6f})",
            }
        )

    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    return fit_results, frontiers


def resolve_extrapolation_range(frontiers, extrapolation_factor, extrapolation_range):
    if extrapolation_range is not None:
        return extrapolation_range

    xs = [comp for pts in frontiers.values() for _, comp, _ in pts]
    if not xs:
        return (1e14, 1e18)
    return (min(xs) / extrapolation_factor, max(xs) * extrapolation_factor)


def save_figure(fig, save_path):
    save_path = ROOT / save_path
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return save_path, pdf_path


def plot_training_curves_by_class(
    experiments,
    show_all_curves=True,
    show_power_law_fit=True,
    show_sklearn_fit=False,
    save_path=DEFAULT_SAVE_PATH,
    classes_to_plot=None,
    flop_range_by_class=None,
    extrapolation_factor=20.0,
    extrapolation_range=None,
    figsize=(12.0, 6.5),
    xlim=None,
    title="x1 Historical Scaling: LayerNorm vs RMSNorm",
    show_legend=True,
    frontier_marker_stride=None,
):
    if show_sklearn_fit:
        print("show_sklearn_fit is accepted for API compatibility but is not implemented in this standalone script.")

    if classes_to_plot is None:
        classes_to_plot = CLASSES

    fit_results, frontiers = fit_and_write_summary(
        experiments,
        classes_to_plot=classes_to_plot,
        flop_range_by_class=flop_range_by_class,
    )
    fit_min, fit_max = resolve_extrapolation_range(
        frontiers,
        extrapolation_factor=extrapolation_factor,
        extrapolation_range=extrapolation_range,
    )

    fig, ax = plt.subplots(figsize=figsize)

    if show_all_curves:
        for name, exp in experiments.items():
            cls = exp.get("class")
            if cls not in classes_to_plot:
                continue
            compute_vals = exp["data"][exp["compute_col"]].values
            loss_vals = exp["data"][exp["loss_col"]].values - IRREDUCIBLE_LOSS
            mask = loss_vals > 0
            if np.any(mask):
                ax.plot(
                    compute_vals[mask],
                    loss_vals[mask],
                    marker=exp["marker"],
                    linestyle=exp["linestyle"],
                    color=exp["color"],
                    alpha=ALPHA_CONFIG["data_points_alpha"],
                    linewidth=1.7,
                    markersize=4,
                )

    for cls in classes_to_plot:
        color = parse_color(CLASS_COLORS[cls])
        label = CLASS_LABELS[cls]
        pts = frontiers[cls]

        if pts:
            sample_stride = frontier_marker_stride or max(1, len(pts) // 18)
            sampled_pts = pts[::sample_stride]
            if sampled_pts[-1] != pts[-1]:
                sampled_pts.append(pts[-1])
            frontier_x = [comp for _, comp, _ in sampled_pts]
            frontier_y = [loss for _, _, loss in sampled_pts]
            ax.scatter(
                frontier_x,
                frontier_y,
                color=color,
                s=70,
                marker="*",
                edgecolors="black",
                linewidth=0.8,
                alpha=0.9,
                zorder=20,
                label=f"{label} frontier",
            )

        if show_power_law_fit:
            params = fit_results.get(cls)
            if params is None:
                continue
            coefficient, exponent, r2 = params
            x_fit = np.logspace(np.log10(fit_min), np.log10(fit_max), 250)
            y_fit = coefficient * np.power(x_fit, exponent)
            ax.plot(
                x_fit,
                y_fit,
                "--",
                linewidth=4,
                alpha=ALPHA_CONFIG["power_law_fit_alpha"],
                color=color,
                label=f"{label} fit: {coefficient:.2e} * C^({exponent:.3f}), R^2={r2:.3f}",
            )

    ax.set_xlabel("Compute (FLOPs)", fontsize=FONT_CONFIG["xlabel_size"])
    ax.set_ylabel(
        "Validation Loss - Irreducible",
        fontsize=FONT_CONFIG["ylabel_size"],
        labelpad=10,
    )
    ax.set_title(
        title,
        fontsize=FONT_CONFIG["title_size"],
        fontweight=FONT_CONFIG["title_weight"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)

    major_formatter = ticker.ScalarFormatter()
    major_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(major_formatter)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)

    ax.tick_params(
        axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"]
    )
    ax.tick_params(
        axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"]
    )

    if show_legend:
        legend = ax.legend(
            loc="upper right",
            fontsize=10,
            framealpha=0.92,
        )
        for handle in legend.legend_handles:
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(1.0)

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.9)
    png_path, pdf_path = save_figure(fig, save_path)
    plt.close(fig)
    return png_path, pdf_path, fit_results, frontiers


def main():
    experiments = load_experiments(build_experiments_config())

    # Single-panel plot of the x1 historical vs x1 RMS historical scaling data.
    # These match the knobs used by the other experimental_analysis single-panel scripts.
    png_path, pdf_path, _, _ = plot_training_curves_by_class(
        experiments,
        show_all_curves=True,
        show_power_law_fit=True,
        show_sklearn_fit=False,  # Accepted for API compatibility; this script fits log-space power laws.
        save_path="Figures/x1_historical_vs_rms_scaling.png",
        classes_to_plot=[
            "x1_historical",
            "x1_rms_historical",
        ],
        flop_range_by_class={
            "x1_historical": (1e16, 5e17),
            "x1_rms_historical": (1e16, 5e17),
        },
        extrapolation_factor=20.0,  # Used when extrapolation_range is None.
        extrapolation_range=(
            10 ** (14.0),
            1e18,
        ),  # Explicitly set extrapolation range (overrides extrapolation_factor).
        xlim=(1e14, 1e18),
    )

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
