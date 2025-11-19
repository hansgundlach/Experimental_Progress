# %%
"""
Script to import TrainingCurveAnalyzer and run plotting sections

This script imports the TrainingCurveAnalyzer class from 'nextgen_lstmvtransformer copy.py'
and loads experiments_config from the same file.
"""

import importlib.util
from pathlib import Path

# Import TrainingCurveAnalyzer from the file with space in name
# Since the filename has a space, we need to use importlib
module_path = Path(__file__).parent / "nextgen_lstmvtransformer copy.py"
spec = importlib.util.spec_from_file_location(
    "nextgen_lstmvtransformer_copy", module_path
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {module_path}")
nextgen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nextgen_module)

# Import the class and constants we need
TrainingCurveAnalyzer = getattr(nextgen_module, "TrainingCurveAnalyzer")
IRREDUCIBLE_LOSS = 1.9  # getattr(nextgen_module, "IRREDUCIBLE_LOSS")

# Configuration
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

# Initialize analyzer
analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping=class_legend_mapping,
)

# Load experiments_config from the same file we loaded the TrainingCurveAnalyzer from
experiments_config_module_path = module_path  # Use the same path we already defined

with open(experiments_config_module_path, "r") as f:
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
        raise ValueError(
            f"Could not find 'experiments_config = [' in {experiments_config_module_path}"
        )

# Fix any typos in the config
if experiments_config:
    for exp in experiments_config:
        if "include_in_in_frontier" in exp:
            exp["include_in_frontier"] = exp.pop("include_in_in_frontier")
else:
    raise ValueError("Failed to extract experiments_config")

# Override include_in_frontier to True for classes that might be plotted
# (The comment says "# Include in frontier analysis" but value is False - this is likely a mistake)
# We'll override to True for common classes that need fits
classes_that_need_fits = [
    "transformer",
    "lstm",
    "sin transformer",
    "sgd",
    "2017 Transformer",
]

if experiments_config:
    for exp in experiments_config:
        exp_class = exp.get("class")
        if exp_class in classes_that_need_fits:
            # If it's explicitly False, override to True so we can compute fits
            if exp.get("include_in_frontier") is False:
                exp["include_in_frontier"] = True
                print(
                    f"Overriding include_in_frontier to True for {exp_class} class experiment: {exp.get('name', 'unknown')}"
                )

# Define which classes we want to plot (used in plot calls)
# Update this list to match the classes you want to plot and fit
classes_to_plot = ["transformer", "lstm"]

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

# Plot training curves
analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    show_sklearn_fit=False,  # Enable sklearn-style fit: L = E + A * C^alpha
    save_path="Figures/transformer_v_lstm_scaling.png",
    classes_to_plot=classes_to_plot,
    flop_range_by_class={
        "transformer": (1e15, 5 * 1e17),
        "lstm": (1e15, 1e17 * 5),
        # "2017 transformer": (1e15, 1e17 * 5),
        # "2017 Transformer": (1e14, 1e17),
        # "sin transformer": (1e16, 1e17 * 5),
        # "sgd": (10 ** (16), 1e17 * 5),
    },
    extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
    # New parameters for explicit control:
    # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
    extrapolation_range=(
        10 ** (14.0),
        1e18,
    ),  # Explicitly set extrapolation range (overrides
)

# %%
# Second plot with different classes
classes_to_plot_2 = ["sin transformer"]

analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    # show_sklearn_fit=True,  # Enable sklearn-style fit: L = E + A * C^alpha
    save_path="Figures/all_ablation_scaling.png",
    classes_to_plot=classes_to_plot_2,
    flop_range_by_class={
        # "transformer": (1e16, 5 * 1e17),
        # "lstm": (1e16, 1e17 * 5),
        # "2017 transformer": (1e15, 1e17 * 5),
        # "2017 Transformer": (1e14, 1e17),
        "sin transformer": (1e15, 1e17 * 5),
        # "sgd": (10 ** (16), 1e17 * 5),
    },
    extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
    # New parameters for explicit control:
    # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
    extrapolation_range=(
        10 ** (14.0),
        1e18,
    ),  # Explicitly set extrapolation range (overrides
    theoretical_scaling_laws=[
        {
            "E": 1.9,  # Irreducible loss
            "A": 83.7,  # Scaling coefficient (larger to be visible)
            "gamma": -0.093,  # Scaling exponent
            "label": "Modern Transformer Fit",
            "color": "purple",
            "linestyle": "--",
        },
    ],
)

#

# %%
classes_to_plot_2 = ["sgd"]

analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    show_sklearn_fit=False,  # Enable sklearn-style fit: L = E + A * C^alpha
    save_path="Figures/transformer_sgd_scaling.png",
    classes_to_plot=classes_to_plot_2,
    flop_range_by_class={
        "transformer": (1e16, 5 * 1e17),
        # "lstm": (1e16, 1e17 * 5),
        # "2017 transformer": (1e15, 1e17 * 5),
        # "2017 Transformer": (1e14, 1e17),
        # "sin transformer": (1e16, 1e17 * 5),
        "sgd": (10 ** (16), 1e17 * 5),
    },
    extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
    # New parameters for explicit control:
    # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
    extrapolation_range=(
        10 ** (14.0),
        1e18,
    ),  # Explicitly set extrapolation range (overrides
    theoretical_scaling_laws=[
        {
            "E": 1.8,  # Irreducible loss
            "A": 83.7,  # Scaling coefficient (larger to be visible)
            "gamma": -0.093,  # Scaling exponent
            "label": "Modern Transformer Fit",
            "color": "purple",
            "linestyle": "--",
        },
    ],
)
# %%

# no-bias comparison
classes_to_plot_2 = ["sin transformer"]
#  Second plot with different classes
# classes_to_plot_2 = ["no_bias","sin transformer"]

analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    # show_sklearn_fit=True,  # Enable sklearn-style fit: L = E + A * C^alpha
    save_path="Figures/no_bias.png",
    classes_to_plot=classes_to_plot_2,
    flop_range_by_class={
        # "transformer": (1e16, 5 * 1e17),
        # "lstm": (1e16, 1e17 * 5),
        # "2017 transformer": (1e15, 1e17 * 5),
        # "2017 Transformer": (1e14, 1e17),
        "sin transformer": (1e16, 1e17 * 5),
        # "no_bias": (1e15, 1e17 * 5),
        # "sgd": (10 ** (16), 1e17 * 5),
    },
    extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
    # New parameters for explicit control:
    # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
    extrapolation_range=(
        10 ** (14.0),
        1e18,
    ),  # Explicitly set extrapolation range (overrides
    theoretical_scaling_laws=[
        {
            "E": 1.9,  # Irreducible loss
            "A": 83.7,  # Scaling coefficient (larger to be visible)
            "gamma": -0.093,  # Scaling exponent
            "label": "Modern Transformer Fit",
            "color": "purple",
            "linestyle": "--",
        },
    ],
)
# %%


# %%

# no-bias comparison
classes_to_plot_2 = ["no_bias", "lstm"]
#  Second plot with different classes
# classes_to_plot_2 = ["no_bias","sin transformer"]

analyzer.plot_training_curves_by_class(
    show_all_curves=True,
    show_power_law_fit=True,
    # show_sklearn_fit=True,  # Enable sklearn-style fit: L = E + A * C^alpha
    save_path="Figures/no_bias.png",
    classes_to_plot=classes_to_plot_2,
    flop_range_by_class={
        # "transformer": (1e16, 5 * 1e17),
        "lstm": (1e16, 1e17 * 5),
        # "2017 transformer": (1e15, 1e17 * 5),
        # "2017 Transformer": (1e14, 1e17),
        # "sin transformer": (1e15, 1e17 * 5),
        "no_bias": (1e16, 1e17 * 5),
        # "sgd": (10 ** (16), 1e17 * 5),
    },
    extrapolation_factor=20.0,  # Extend trend lines 3x beyond data range
    # New parameters for explicit control:
    # xlim=(1e13, 1e18),  # Explicitly set x-axis limits from 10^14 to 10^18
    extrapolation_range=(
        10 ** (14.0),
        1e18,
    ),  # Explicitly set extrapolation range (overrides
    theoretical_scaling_laws=[
        {
            "E": 1.9,  # Irreducible loss
            "A": 83.7,  # Scaling coefficient (larger to be visible)
            "gamma": -0.093,  # Scaling exponent
            "label": "Modern Transformer Fit",
            "color": "purple",
            "linestyle": "--",
        },
    ],
)
# %%
