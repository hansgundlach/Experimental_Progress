# Training Curve Analyzer

A modular system for analyzing and plotting training curves from multiple CSV files, with automatic frontier identification and power law fitting.

## Features

- **Arbitrary CSV Loading**: Load any number of CSV files with training data
- **Frontier Identification**: Automatically identify Pareto frontier points (best performing experiments)
- **Power Law Fitting**: Fit scaling laws to frontier points
- **Flexible Plotting**: Multiple plotting options with customizable styling
- **Error Handling**: Robust error handling for missing files or invalid data

## Quick Start

```python
from lstm_scaling import TrainingCurveAnalyzer

# Initialize analyzer
analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)

# Add experiments
analyzer.add_experiment(
    name='My Experiment',
    csv_path='path/to/experiment.csv',
    color='red',
    marker='o'
)

# Identify frontier
analyzer.identify_frontier(method='pareto')

# Plot results
analyzer.plot_training_curves(
    show_all_curves=True,
    show_power_law_fit=True,
    save_path='my_plot.png'
)
```

## CSV File Requirements

Your CSV files should contain at least these columns:

- `total_flops_profiler` (or custom compute column): Compute values (FLOPS)
- `validation_loss` (or custom loss column): Validation loss values

## Methods

### `add_experiment(name, csv_path, **kwargs)`

Add an experiment from a CSV file.

**Parameters:**

- `name` (str): Name for the experiment
- `csv_path` (str): Path to the CSV file
- `compute_col` (str, optional): Column name for compute values (default: 'total_flops_profiler')
- `loss_col` (str, optional): Column name for loss values (default: 'validation_loss')
- `color` (str, optional): Color for plotting (auto-assigned if None)
- `marker` (str, optional): Marker style (default: 'o')
- `linestyle` (str, optional): Line style (default: '-')
- `alpha` (float, optional): Transparency (default: 0.6)
- `include_in_frontier` (bool, optional): Whether to include this experiment in frontier analysis (default: True)

### `identify_frontier(method='pareto')`

Identify frontier points (best performing experiments).

**Parameters:**

- `method` (str): 'pareto' (Pareto frontier) or 'top_n' (top N by loss)

**Returns:**

- List of experiment names on the frontier

### `fit_power_law(experiment_names=None)`

Fit a power law to specified experiments.

**Parameters:**

- `experiment_names` (list, optional): List of experiment names (uses frontier if None)

**Returns:**

- Tuple of (a, b) parameters for power law y = a * x^b, or None if fitting fails

### `plot_training_curves(**kwargs)`

Plot training curves for experiments.

**Parameters:**

- `show_all_curves` (bool): Whether to show all training curves (default: True)
- `show_frontier_only` (bool): Whether to only show frontier experiments (default: False)
- `show_power_law_fit` (bool): Whether to show power law fit (default: True)
- `figsize` (tuple): Figure size (default: (12, 8))
- `save_path` (str, optional): Path to save the plot

## Frontier Identification Methods

### Pareto Frontier

Finds experiments that are not dominated by any other experiment. An experiment dominates another if it has:

- Lower or equal compute AND lower loss, OR
- Lower compute AND lower or equal loss

### Top N

Selects the N experiments with the lowest final loss values.

## Example Usage

See `example_usage.py` for complete examples.

### Basic Usage

```python
# Load multiple experiments
experiments = [
    {'name': '16d', 'csv_path': '16d.csv', 'color': 'orange'},
    {'name': '32d', 'csv_path': '32d.csv', 'color': 'blue'},
    {'name': '64d', 'csv_path': '64d.csv', 'color': 'green'},
]

analyzer = TrainingCurveAnalyzer(irreducible_loss=1.76)
for exp in experiments:
    analyzer.add_experiment(**exp)

# Find frontier and plot
analyzer.identify_frontier()
analyzer.plot_training_curves()
```

### Custom Column Names

```python
analyzer.add_experiment(
    name='Custom Experiment',
    csv_path='experiment.csv',
    compute_col='flops',  # Custom compute column
    loss_col='val_loss',  # Custom loss column
    color='red',
    marker='*'
)
```

### Frontier Exclusion

You can exclude certain experiments from frontier analysis while still plotting them:

```python
# Main experiments (included in frontier analysis)
analyzer.add_experiment(
    name='Main Experiment',
    csv_path='main.csv',
    color='blue',
    include_in_frontier=True  # Will be considered for frontier
)

# Baseline experiments (excluded from frontier analysis)
analyzer.add_experiment(
    name='Baseline',
    csv_path='baseline.csv',
    color='gray',
    include_in_frontier=False  # Won't be considered for frontier
)
```

### Plotting Options

```python
# Plot only frontier experiments
analyzer.plot_training_curves(
    show_all_curves=False,
    show_frontier_only=True,
    show_power_law_fit=True
)

# Plot with custom figure size
analyzer.plot_training_curves(
    figsize=(15, 10),
    save_path='large_plot.png'
)
```

## Output

The system produces:

1. **Console output**: Information about loaded experiments and frontier identification
2. **Plots**: Training curves with frontier points highlighted (starred markers)
3. **Power law fits**: Fitted scaling laws for frontier points

## Error Handling

The system handles:

- Missing CSV files
- Invalid column names
- Insufficient data for power law fitting
- Invalid file paths

All errors are reported to the console without crashing the analysis.
