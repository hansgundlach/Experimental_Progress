"""Compute scaling law for x1_all_lr_historical_trend vs x1_historical."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"
FIGURES_DIR = ROOT / "Figures"
SUMMARY_PATH = Path(__file__).resolve().parent / "x1_all_lr_historical_trend_summary.csv"

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

# 128d excluded from trend — diverged (all NaN)
DIMS_TREND = [16, 32, 48, 64, 80, 96, 112, 160, 192, 224, 256]
DIMS_HIST  = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]

CLASSES = ["x1_all_lr_historical_trend", "x1_historical"]

CLASS_LABELS = {
    "x1_all_lr_historical_trend": "Historical Transformer (trend LR)",
    "x1_historical": "Historical Transformer (per-dim LR)",
}

CLASS_COLORS = {
    "x1_all_lr_historical_trend": "viridis[0.2]",
    "x1_historical": "plasma[0.70]",
}

DEFAULT_SAVE_PATH = "Figures/x1_all_lr_historical_trend_scaling.png"


def parse_color(color_spec):
    if "[" not in color_spec or not color_spec.endswith("]"):
        return color_spec
    cmap_name, raw_index = color_spec[:-1].split("[", 1)
    return plt.get_cmap(cmap_name)(float(raw_index))


def build_experiments_config():
    configs = []
    for dim in DIMS_TREND:
        csv_path = DATA_ROOT / "x1_all_lr_historical_trend" / f"{dim}d.csv"
        configs.append({
            "name": f"{dim}d trend-LR",
            "csv_path": str(csv_path),
            "class": "x1_all_lr_historical_trend",
            "hidden_dim": dim,
            "color": CLASS_COLORS["x1_all_lr_historical_trend"],
        })
    for dim in DIMS_HIST:
        csv_path = DATA_ROOT / "x1_historical" / f"{dim}_all_reset.csv"
        if not csv_path.exists():
            continue
        configs.append({
            "name": f"{dim}d per-dim-LR",
            "csv_path": str(csv_path),
            "class": "x1_historical",
            "hidden_dim": dim,
            "color": CLASS_COLORS["x1_historical"],
        })
    return configs


def load_experiments(configs):
    experiments = {}
    for config in configs:
        path = Path(config["csv_path"])
        if not path.exists():
            print(f"  SKIP (missing): {path}")
            continue
        df = pd.read_csv(path)
        if df["validation_loss"].dropna().empty:
            print(f"  SKIP (all NaN): {path.name}")
            continue
        experiments[config["name"]] = {
            "data": df,
            "compute_col": "total_flops_profiler",
            "loss_col": "validation_loss",
            "color": parse_color(config["color"]),
            "marker": "o",
            "linestyle": "-",
            "class": config["class"],
            "hidden_dim": config["hidden_dim"],
        }
    return experiments


def get_frontier_points(experiments, cls, min_compute=None, max_compute=None):
    points = []
    for name, exp in experiments.items():
        if exp["class"] != cls:
            continue
        compute_vals = exp["data"][exp["compute_col"]].to_numpy(dtype=float)
        loss_vals = exp["data"][exp["loss_col"]].to_numpy(dtype=float) - IRREDUCIBLE_LOSS
        mask = loss_vals > 0
        if min_compute is not None:
            mask &= compute_vals >= min_compute
        if max_compute is not None:
            mask &= compute_vals <= max_compute
        points.extend(
            (name, float(c), float(l))
            for c, l in zip(compute_vals[mask], loss_vals[mask])
        )
    frontier = []
    best_loss = float("inf")
    for name, compute, loss in sorted(points, key=lambda p: p[1]):
        if loss < best_loss:
            frontier.append((name, compute, loss))
            best_loss = loss
    return frontier


def fit_power_law(frontier):
    x = np.array([p[1] for p in frontier], dtype=float)
    y = np.array([p[2] for p in frontier], dtype=float)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    coeff = float(np.exp(intercept))
    exp_ = float(slope)
    y_pred = coeff * x ** exp_
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return coeff, exp_, r2


def fit_and_write_summary(experiments, classes_to_plot, flop_range_by_class=None):
    frontiers = {}
    fit_results = {}
    for cls in classes_to_plot:
        lo, hi = (None, None)
        if flop_range_by_class and cls in flop_range_by_class:
            lo, hi = flop_range_by_class[cls]
        frontiers[cls] = get_frontier_points(experiments, cls, lo, hi)
        fit_results[cls] = fit_power_law(frontiers[cls]) if len(frontiers[cls]) >= 2 else None

    rows = []
    for cls in classes_to_plot:
        params = fit_results.get(cls)
        if params is None:
            continue
        coeff, exp_, r2 = params
        rows.append({
            "class": cls,
            "label": CLASS_LABELS[cls],
            "coefficient": coeff,
            "scaling_exponent": exp_,
            "r_squared": r2,
            "frontier_points": len(frontiers[cls]),
            "law": f"{coeff:.6e} * C^({exp_:.6f})",
        })
        print(f"  {CLASS_LABELS[cls]}: L-E = {coeff:.4e} * C^({exp_:.4f}),  R²={r2:.4f}")

    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    return fit_results, frontiers


def save_figure(fig, save_path):
    save_path = ROOT / save_path
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return save_path, pdf_path


def plot(
    experiments,
    save_path=DEFAULT_SAVE_PATH,
    classes_to_plot=None,
    flop_range_by_class=None,
    extrapolation_range=(1e14, 1e18),
    xlim=(1e14, 1e18),
    title="Compute scaling: Historical Transformer — trend LR vs per-dim LR",
):
    if classes_to_plot is None:
        classes_to_plot = CLASSES

    print("\nPower-law fits (L − E = A · C^α):")
    fit_results, frontiers = fit_and_write_summary(
        experiments, classes_to_plot, flop_range_by_class
    )

    fit_min, fit_max = extrapolation_range
    fig, ax = plt.subplots(figsize=(12.0, 6.5))

    # Training curves
    for name, exp in experiments.items():
        cls = exp["class"]
        if cls not in classes_to_plot:
            continue
        compute_vals = exp["data"][exp["compute_col"]].values
        loss_vals = exp["data"][exp["loss_col"]].values - IRREDUCIBLE_LOSS
        mask = loss_vals > 0
        if np.any(mask):
            ax.plot(
                compute_vals[mask], loss_vals[mask],
                marker=exp["marker"], linestyle=exp["linestyle"],
                color=exp["color"],
                alpha=ALPHA_CONFIG["data_points_alpha"],
                linewidth=1.7, markersize=4,
            )

    # Frontier stars + fit lines
    for cls in classes_to_plot:
        color = parse_color(CLASS_COLORS[cls])
        label = CLASS_LABELS[cls]
        pts = frontiers[cls]

        if pts:
            stride = max(1, len(pts) // 18)
            sampled = pts[::stride]
            if sampled[-1] != pts[-1]:
                sampled.append(pts[-1])
            ax.scatter(
                [p[1] for p in sampled], [p[2] for p in sampled],
                color=color, s=70, marker="*",
                edgecolors="black", linewidth=0.8,
                alpha=0.9, zorder=20, label=f"{label} frontier",
            )

        params = fit_results.get(cls)
        if params is not None:
            coeff, exp_, r2 = params
            x_fit = np.logspace(np.log10(fit_min), np.log10(fit_max), 250)
            y_fit = coeff * x_fit ** exp_
            ax.plot(
                x_fit, y_fit, "--", linewidth=4,
                alpha=ALPHA_CONFIG["power_law_fit_alpha"], color=color,
                label=f"{label} fit: {coeff:.2e}·C^({exp_:.3f}), R²={r2:.3f}",
            )

    ax.set_xlabel("Compute (FLOPs)", fontsize=FONT_CONFIG["xlabel_size"])
    ax.set_ylabel("Validation Loss − Irreducible", fontsize=FONT_CONFIG["ylabel_size"], labelpad=10)
    ax.set_title(title, fontsize=FONT_CONFIG["title_size"], fontweight=FONT_CONFIG["title_weight"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)

    fmt = ticker.ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.tick_params(axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"])

    legend = ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.9)
    png_path, pdf_path = save_figure(fig, save_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    experiments = load_experiments(build_experiments_config())

    png_path, pdf_path = plot(
        experiments,
        save_path=DEFAULT_SAVE_PATH,
        classes_to_plot=CLASSES,
        flop_range_by_class={
            "x1_all_lr_historical_trend": (1e14, 5e17),
            "x1_historical": (1e14, 5e17),
        },
        extrapolation_range=(1e14, 1e18),
        xlim=(1e14, 1e18),
    )
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
