"""Single-panel compute scaling figure for LSTM layer-2 trend-LR experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphing_utils import TrainingCurveAnalyzer, ALPHA_CONFIG

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"

IRREDUCIBLE_LOSS = 1.9
DIMS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 512]
FLOP_RANGE = (1e16, 5e17)
EXTRAPOLATION_RANGE = (1e13, 1e18)

FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 14,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 12,
    "minor_tick_size": 10,
}


def main():
    analyzer = TrainingCurveAnalyzer(
        irreducible_loss=IRREDUCIBLE_LOSS,
        use_theoretical_flops=False,
        class_legend_mapping={"lstm_layer2": "LSTM (2 layers)"},
    )

    for dim in DIMS:
        path = DATA_ROOT / "x2_lstm_layer2_lr_trend" / f"{dim}d.csv"
        if not path.exists():
            print(f"  SKIP (missing): {dim}d.csv")
            continue
        analyzer.add_experiment(
            name=f"{dim}d layer2",
            csv_path=str(path),
            color="cividis[0.65]",
            marker="o",
            include_in_frontier=True,
            class_name="lstm_layer2",
            hidden_dim=dim,
        )

    analyzer.identify_frontier_by_class(
        method="pareto", flop_range=None, use_all_points=True,
        classes=["lstm_layer2"],
        flop_range_by_class={"lstm_layer2": FLOP_RANGE},
    )
    fit = analyzer.fit_power_law_by_class(class_names=["lstm_layer2"], use_all_points=True)
    a, b, r2 = fit["lstm_layer2"]
    print(f"LSTM Layer 2: L-E = {a:.4f}·C^({b:.4f}), R²={r2:.4f}")

    color = analyzer.get_class_color("lstm_layer2")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    plotted = False
    for name, exp in analyzer.experiments.items():
        cv = exp["data"][exp["compute_col"]].values
        lv = exp["data"][exp["loss_col"]].values - IRREDUCIBLE_LOSS
        mask = lv > 0
        if np.any(mask):
            ax.plot(cv[mask], lv[mask], color=color,
                    alpha=ALPHA_CONFIG["data_points_alpha"],
                    linewidth=1.5, markersize=3, marker="o", linestyle="-",
                    label="LSTM (2 layers)" if not plotted else None)
            plotted = True

    x_fit = np.logspace(np.log10(EXTRAPOLATION_RANGE[0]), np.log10(EXTRAPOLATION_RANGE[1]), 200)
    ax.plot(x_fit, a * x_fit ** b, "--", linewidth=3, color=color,
            alpha=ALPHA_CONFIG["power_law_fit_alpha"],
            label=f"Fit: {a:.2e}·C^({b:.3f}), R²={r2:.3f}")

    ax.text(0.03, 0.04, f"A={a:.2e}, α={b:.3f}, R²={r2:.3f}",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.88})

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs)", fontsize=FONT_CONFIG["xlabel_size"])
    ax.set_ylabel("Validation Loss − Irreducible", fontsize=FONT_CONFIG["ylabel_size"])
    ax.set_title("LSTM 2 Layer Scaling",
                 fontsize=FONT_CONFIG["title_size"], fontweight=FONT_CONFIG["title_weight"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9, prop={"weight": "bold"})
    ax.tick_params(axis="both", which="major", labelsize=FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=FONT_CONFIG["minor_tick_size"])

    plt.tight_layout()

    out = ROOT / "Figures" / "lstm_layer2_lr_trend_scaling.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
