"""
Standalone script to generate the SGD Scaling Experiments single-panel plot.

Data:  sgd_scaling/data/*.csv  (copied from experimental_data_folder/x1_new_sgd_scaling/)
Output: sgd_scaling/figures/transformer_sgd_scaling.png  (and .pdf)

Run from *this* directory:
    cd sgd_scaling
    python generate_plot.py
"""

import sys
from pathlib import Path

# Make graphing_utils importable from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphing_utils import TrainingCurveAnalyzer

IRREDUCIBLE_LOSS = 1.9
USE_THEORETICAL_FLOPS = False   # use total_flops_profiler column

# ---------------------------------------------------------------------------
# Experiment definitions (SGD class only)
# ---------------------------------------------------------------------------
SGD_EXPERIMENTS = [
    {"name": "32d SGD",  "csv_path": "data/32d_sgdbs64.csv",  "hidden_dim": 32},
    {"name": "48d SGD",  "csv_path": "data/48d_sgdbs64.csv",  "hidden_dim": 48},
    {"name": "64d SGD",  "csv_path": "data/64d_sgdbs64.csv",  "hidden_dim": 64},
    {"name": "96d SGD",  "csv_path": "data/96d_sgdbs64.csv",  "hidden_dim": 96},
    {"name": "128d SGD", "csv_path": "data/128d_sgdbs64.csv", "hidden_dim": 128},
    {"name": "160d SGD", "csv_path": "data/160d_sgdbs64.csv", "hidden_dim": 160},
    {"name": "192d SGD", "csv_path": "data/192d_sgdbs64.csv", "hidden_dim": 192},
    {"name": "256d SGD", "csv_path": "data/256d_sgdbs64.csv", "hidden_dim": 256},
]

# ---------------------------------------------------------------------------
# Build analyzer
# ---------------------------------------------------------------------------
analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping={"sgd": "SGD experiments"},
)

for exp in SGD_EXPERIMENTS:
    analyzer.add_experiment(
        name=exp["name"],
        csv_path=exp["csv_path"],
        marker="o",
        include_in_frontier=True,
        class_name="sgd",
        hidden_dim=exp["hidden_dim"],
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    show_sklearn_fit=False,
    save_path=str(figures_dir / "transformer_sgd_scaling.png"),
    title="SGD Scaling Experiments",
    classes_to_plot=["sgd"],
    flop_range_by_class={
        "sgd": (10**16, 1e17 * 5),
    },
    extrapolation_factor=20.0,
    extrapolation_range=(10**14.0, 1e18),
    theoretical_scaling_laws=[
        {
            "E": 1.9,
            "A": 85.6,
            "gamma": -0.093,
            "label": "Modern Transformer Fit",
            "color": "purple",
            "linestyle": "--",
            "show_irreducible_loss": False,
        },
    ],
)

print("Done — figure saved to figures/transformer_sgd_scaling.png")
