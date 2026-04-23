"""Mixed single-panel scaling plot: x1_lstm_layer1 for hidden_dim < 256, x2_lstm_layer2 for hidden_dim >= 256."""

from graphing_utils import TrainingCurveAnalyzer, IRREDUCIBLE_LOSS

# Experiment configs: layer1 for dims < 256, layer2 for dims >= 256
experiments_config = [
    # ---- x1_lstm_layer1 (hidden_dim < 256) ----
    {
        "name": "32d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/32d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 32,
    },
    {
        "name": "48d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/48d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 48,
    },
    {
        "name": "64d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/64d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 64,
    },
    {
        "name": "128d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/128d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 128,
    },
    {
        "name": "160d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/160d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 160,
    },
    {
        "name": "192d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/192d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 192,
    },
    {
        "name": "224d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/224d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 224,
    },
    # ---- x2_lstm_layer2 (hidden_dim >= 256) ----
    {
        "name": "256d x2 lstm layer2",
        "csv_path": "../new_experiments_folder_1/x2_lstm_layer2/256d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 256,
    },
    {
        "name": "384d x2 lstm layer2",
        "csv_path": "../new_experiments_folder_1/x2_lstm_layer2/384d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 384,
    },
    {
        "name": "448d x2 lstm layer2",
        "csv_path": "../new_experiments_folder_1/x2_lstm_layer2/448d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 448,
    },
    {
        "name": "512d x2 lstm layer2",
        "csv_path": "../new_experiments_folder_1/x2_lstm_layer2/512d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 512,
    },
]


if __name__ == "__main__":
    USE_THEORETICAL_FLOPS = (
        False  # Set to True to use theoretical_flops, False for total_flops_profiler
    )

    # Define class-to-legend-label mapping for cleaner legend
    class_legend_mapping = {
        "lstm": "LSTM (1-layer < 256d, 2-layer >= 256d)",
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

    # Mixed single-panel plot
    analyzer.plot_training_curves_by_class(
        show_all_curves=True,
        show_power_law_fit=True,
        show_sklearn_fit=False,
        save_path="Figures/lstm_mixed_layer1_layer2_scaling.png",
        classes_to_plot=[
            "lstm",
        ],
        flop_range_by_class={
            "lstm": (1e15, 5 * 1e17),
        },
        extrapolation_factor=20.0,
        extrapolation_range=(
            10 ** (14.0),
            1e18,
        ),
    )

# %%
