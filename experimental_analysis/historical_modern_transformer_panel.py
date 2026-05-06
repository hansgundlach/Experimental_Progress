"""Single-panel scaling figure for historical, RMS historical, and modern transformer runs."""

from datetime import datetime
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphing_utils import ALPHA_CONFIG, TrainingCurveAnalyzer


IRREDUCIBLE_LOSS = 1.9
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"

FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 14,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 12,
    "minor_tick_size": 10,
    "legend_size": 11,
}

DIMS = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 352]
FLOP_RANGE = (1e14, 5e17)
EXTRAPOLATION_RANGE = (1e14, 1e18)


def add_dim_sweep(configs, *, class_name, label, folder, filename_for_dim, color):
    for dim in DIMS:
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


def build_analyzer():
    experiments_config = []
    add_dim_sweep(
        experiments_config,
        class_name="historical_transformer",
        label="Historical Transformer",
        folder="x1_historical",
        filename_for_dim=lambda dim: f"{dim}_all_reset.csv",
        color="plasma[0.70]",
    )
    add_dim_sweep(
        experiments_config,
        class_name="historical_rms_transformer",
        label="Historical Transformer + RMSNorm",
        folder="x1_rms_historical",
        filename_for_dim=lambda dim: f"{dim}_all_reset.csv",
        color="viridis[0.55]",
    )
    add_dim_sweep(
        experiments_config,
        class_name="modern_transformer",
        label="Modern Transformer",
        folder="x2_modern",
        filename_for_dim=lambda dim: (
            f"{dim}_modern.csv" if dim in (48, 112) else f"{dim}_modern_40.csv"
        ),
        color="viridis[0.15]",
    )

    analyzer = TrainingCurveAnalyzer(
        irreducible_loss=IRREDUCIBLE_LOSS,
        use_theoretical_flops=False,
        class_legend_mapping={
            "historical_transformer": "Historical Transformer",
            "historical_rms_transformer": "Historical Transformer + RMSNorm",
            "modern_transformer": "Modern Transformer",
        },
    )

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

    return analyzer


def plot_panel_to_axis(analyzer, ax, class_names):
    analyzer.identify_frontier_by_class(
        method="pareto",
        flop_range=None,
        use_all_points=True,
        classes=class_names,
        flop_range_by_class={class_name: FLOP_RANGE for class_name in class_names},
    )

    plotted_classes = set()

    for _, exp in analyzer.experiments.items():
        class_name = exp.get("class")
        if class_name not in class_names:
            continue

        label = analyzer.class_legend_mapping.get(class_name, class_name)
        compute_vals = exp["data"][exp["compute_col"]].values
        loss_vals = exp["data"][exp["loss_col"]].values - analyzer.irreducible_loss
        mask = loss_vals > 0
        if not np.any(mask):
            continue

        ax.plot(
            compute_vals[mask],
            loss_vals[mask],
            marker=exp["marker"],
            linestyle=exp["linestyle"],
            color=exp["color"],
            alpha=ALPHA_CONFIG["data_points_alpha"],
            label=label if class_name not in plotted_classes else None,
            linewidth=1.5,
            markersize=3,
        )
        plotted_classes.add(class_name)

    fit_results = analyzer.fit_power_law_by_class(
        class_names=class_names,
        use_all_points=True,
    )
    fit_summaries = []
    for class_name in class_names:
        params = fit_results.get(class_name)
        if params is None:
            continue

        label = analyzer.class_legend_mapping.get(class_name, class_name)
        coefficient, exponent, r2 = params
        fit_summaries.append(
            f"{label}: A={coefficient:.2e}, alpha={exponent:.3f}, R^2={r2:.3f}"
        )
        x_fit = np.logspace(
            np.log10(EXTRAPOLATION_RANGE[0]),
            np.log10(EXTRAPOLATION_RANGE[1]),
            200,
        )
        y_fit = coefficient * np.power(x_fit, exponent)
        ax.plot(
            x_fit,
            y_fit,
            "--",
            linewidth=3,
            alpha=ALPHA_CONFIG["power_law_fit_alpha"],
            label=f"{label} fit:\n{coefficient:.2e} * C^({exponent:.3f}), R^2={r2:.3f}",
            color=analyzer.get_class_color(class_name),
        )

    ax.set_xlabel("Compute (FLOPs)", fontsize=FONT_CONFIG["xlabel_size"])
    ax.set_ylabel(
        "Validation Loss - Irreducible",
        fontsize=FONT_CONFIG["ylabel_size"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"])
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
    ax.legend(
        loc="upper right",
        framealpha=0.9,
        prop={"weight": "bold", "size": FONT_CONFIG["legend_size"]},
    )


def main():
    analyzer = build_analyzer()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_to_axis(
        analyzer,
        ax,
        [
            "historical_transformer",
            "historical_rms_transformer",
            "modern_transformer",
        ],
    )

    fig.suptitle(
        "Historical, RMS Historical, and Modern Transformer Scaling",
        fontsize=FONT_CONFIG["title_size"] + 2,
        fontweight=FONT_CONFIG["title_weight"],
        y=0.98,
    )

    plt.tight_layout()

    png_path = ROOT / "Figures" / "historical_rms_modern_transformer_same_panel.png"
    pdf_path = ROOT / "Figures" / "historical_rms_modern_transformer_same_panel.pdf"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_png_path = (
        ROOT / "Figures" / f"historical_rms_modern_transformer_same_panel_{timestamp}.png"
    )
    timestamped_pdf_path = (
        ROOT / "Figures" / f"historical_rms_modern_transformer_same_panel_{timestamp}.pdf"
    )
    png_path.parent.mkdir(exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(timestamped_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(timestamped_pdf_path, dpi=300, bbox_inches="tight")

    for path in (png_path, pdf_path, timestamped_png_path, timestamped_pdf_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"Saved {path.resolve()} sha256={digest}")


if __name__ == "__main__":
    main()
