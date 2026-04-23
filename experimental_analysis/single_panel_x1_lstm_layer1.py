"""Single-panel scaling plot for the new x1_lstm_layer1 data in new_experiments_folder_1."""

from graphing_utils import TrainingCurveAnalyzer, IRREDUCIBLE_LOSS

# Experiment configs for the new x1_lstm_layer1 run
experiments_config = [
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
    {
        "name": "256d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/256d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 256,
    },
    {
        "name": "320d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/320d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 320,
    },
    {
        "name": "384d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/384d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 384,
    },
    {
        "name": "448d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/448d.csv",
        "marker": "o",
        "include_in_frontier": True,
        "class": "lstm",
        "hidden_dim": 448,
    },
    {
        "name": "512d x1 lstm layer1",
        "csv_path": "../new_experiments_folder_1/x1_lstm_layer1/512d.csv",
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
        "lstm": "LSTM (1 layer, x1)",
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

    # Single-panel plot of the x1 LSTM layer-1 scaling data
    analyzer.plot_training_curves_by_class(
        show_all_curves=True,
        show_power_law_fit=True,
        show_sklearn_fit=False,  # Enable sklearn-style fit: L = E + A * C^alpha
        save_path="Figures/x1_lstm_layer1_scaling.png",
        classes_to_plot=[
            "lstm",
        ],
        flop_range_by_class={
            "lstm": (1e15, 5 * 1e17),
        },
        extrapolation_factor=20.0,  # Extend trend lines 20x beyond data range
        extrapolation_range=(
            10 ** (14.0),
            1e18,
        ),  # Explicitly set extrapolation range (overrides extrapolation_factor)
    )

# %%
