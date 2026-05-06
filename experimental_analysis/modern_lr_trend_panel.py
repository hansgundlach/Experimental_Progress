"""Compute scaling panel for x1_modern_all_lr_trend (modern transformer, trend LR)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"
SUMMARY_PATH = Path(__file__).resolve().parent / "modern_lr_trend_summary.csv"

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

DIMS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
CLASS = "x1_modern_all_lr_trend"
CLASS_LABEL = "Modern Transformer (trend LR)"
CLASS_COLOR = "plasma[0.35]"
DEFAULT_SAVE_PATH = "Figures/modern_all_lr_trend_scaling.png"


def parse_color(color_spec):
    if "[" not in color_spec or not color_spec.endswith("]"):
        return color_spec
    cmap_name, raw_index = color_spec[:-1].split("[", 1)
    return plt.get_cmap(cmap_name)(float(raw_index))


def load_experiments():
    experiments = {}
    for dim in DIMS:
        path = DATA_ROOT / "x1_modern_all_lr_trend" / f"{dim}d.csv"
        if not path.exists():
            print(f"  SKIP (missing): {path.name}")
            continue
        df = pd.read_csv(path)
        valid = df["validation_loss"].dropna()
        if valid.empty:
            print(f"  SKIP (all NaN): {path.name}")
            continue
        print(f"  {dim}d: final_loss={valid.iloc[-1]:.4f}, compute={df['total_flops_profiler'].iloc[-1]:.2e}")
        experiments[f"{dim}d"] = {
            "data": df,
            "compute_col": "total_flops_profiler",
            "loss_col": "validation_loss",
            "color": parse_color(CLASS_COLOR),
            "dim": dim,
        }
    return experiments


def get_frontier(experiments, min_compute=None, max_compute=None):
    points = []
    for name, exp in experiments.items():
        compute = exp["data"][exp["compute_col"]].to_numpy(dtype=float)
        loss = exp["data"][exp["loss_col"]].to_numpy(dtype=float) - IRREDUCIBLE_LOSS
        mask = loss > 0
        if min_compute is not None:
            mask &= compute >= min_compute
        if max_compute is not None:
            mask &= compute <= max_compute
        points.extend((float(c), float(l)) for c, l in zip(compute[mask], loss[mask]))

    frontier = []
    best = float("inf")
    for c, l in sorted(points):
        if l < best:
            frontier.append((c, l))
            best = l
    return frontier


def fit_power_law(frontier):
    x = np.array([p[0] for p in frontier], dtype=float)
    y = np.array([p[1] for p in frontier], dtype=float)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    coeff = float(np.exp(intercept))
    exp_ = float(slope)
    y_pred = coeff * x ** exp_
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return coeff, exp_, r2


def main():
    print("Loading experiments...")
    experiments = load_experiments()
    if not experiments:
        print("No data available yet — re-run when jobs complete.")
        return

    frontier = get_frontier(experiments, min_compute=1e14, max_compute=5e17)
    coeff, exp_, r2 = fit_power_law(frontier)
    print(f"\nPower-law fit (L − E = A · C^α):")
    print(f"  {CLASS_LABEL}: L-E = {coeff:.4e} · C^({exp_:.4f}),  R²={r2:.4f}")

    pd.DataFrame([{
        "class": CLASS, "label": CLASS_LABEL,
        "coefficient": coeff, "scaling_exponent": exp_, "r_squared": r2,
        "frontier_points": len(frontier),
        "law": f"{coeff:.6e} * C^({exp_:.6f})",
    }]).to_csv(SUMMARY_PATH, index=False)

    color = parse_color(CLASS_COLOR)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for name, exp in experiments.items():
        compute = exp["data"][exp["compute_col"]].values
        loss = exp["data"][exp["loss_col"]].values - IRREDUCIBLE_LOSS
        mask = loss > 0
        if np.any(mask):
            ax.plot(compute[mask], loss[mask],
                    color=color, alpha=ALPHA_CONFIG["data_points_alpha"],
                    linewidth=1.7, markersize=4, marker="o", linestyle="-")

    # frontier stars
    stride = max(1, len(frontier) // 18)
    sampled = frontier[::stride]
    if sampled[-1] != frontier[-1]:
        sampled = sampled + [frontier[-1]]
    ax.scatter([p[0] for p in sampled], [p[1] for p in sampled],
               color=color, s=70, marker="*", edgecolors="black",
               linewidth=0.8, alpha=0.9, zorder=20,
               label=f"{CLASS_LABEL} frontier")

    x_fit = np.logspace(14, 18, 250)
    ax.plot(x_fit, coeff * x_fit ** exp_, "--", linewidth=4,
            alpha=ALPHA_CONFIG["power_law_fit_alpha"], color=color,
            label=f"{CLASS_LABEL} fit:\n{coeff:.2e}·C^({exp_:.3f}), R²={r2:.3f}")

    ax.set_xlabel("Compute (FLOPs)", fontsize=FONT_CONFIG["xlabel_size"])
    ax.set_ylabel("Validation Loss − Irreducible", fontsize=FONT_CONFIG["ylabel_size"], labelpad=10)
    ax.set_title("Modern Transformer Scaling (Trend LR, 16d–256d)",
                 fontsize=FONT_CONFIG["title_size"], fontweight=FONT_CONFIG["title_weight"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e14, 1e18)
    ax.grid(True, alpha=0.3)

    fmt = ticker.ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.tick_params(axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"])

    ax.text(0.03, 0.04,
            f"A={coeff:.2e}, α={exp_:.3f}, R²={r2:.3f}",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.88})
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.9)
    save_path = ROOT / DEFAULT_SAVE_PATH
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")
    print(f"Saved: {save_path.with_suffix('.pdf')}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
