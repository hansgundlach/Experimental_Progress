"""Standalone single-panel plotting script preserved from the original nextgen_lstmvtransformer file."""

from graphing_utils import TrainingCurveAnalyzer, FONT_CONFIG, ALPHA_CONFIG, IRREDUCIBLE_LOSS, CLASS_COLORS
from experiments_config import experiments_config

if __name__ == "__main__":
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
        "sin transformer": "2017 Transformer",
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
        save_path="Figures/transformer_v_lstm_scaling.png",
        classes_to_plot=[
            # "optimal_lr_sgd_transformer",
            "transformer",
            "lstm",
            # "sin transformer",
            # "2017 Transformer",
            # "sgd",
            # "vanilla_transformer_rmsprop",
        ],
        flop_range_by_class={
            "transformer": (1e15, 5 * 1e17),
            "lstm": (1e15, 1e17 * 5),
            # "2017 transformer": (1e15, 1e17 * 5),
            # "2017 Transformer": (1e14, 1e17),
            # "sgd": (1e14, 1e17),
        },
        extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
        # New parameters for explicit control:
        # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
        extrapolation_range=(
            10 ** (14.0),
            1e18,
        ),  # Explicitly set extrapolation range (overrides extrapolation_factor)
        # Example theoretical scaling laws to compare with data
        theoretical_scaling_laws=[
            {
                "E": 1.8,  # Irreducible loss
                "A": 76.6,  # Scaling coefficient (larger to be visible)
                "gamma": -0.090,  # Scaling exponent
                "label": "Modern Transformer Fit",
                "color": "purple",
                "linestyle": "--",
                "linewidth": 3,
                "linewidth": 3,
                "alpha": 0.8,
            },
        ],
    )

# %%
