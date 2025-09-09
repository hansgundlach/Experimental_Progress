# LSTM Experiment Structure

This directory contains the LSTM experiment framework with a modular structure for experiment definitions.

## Files

### `lstm_experiment_definitions.py`

Contains all experiment definitions and utility functions for creating experiment variations. This file includes:

- **Experiment Definitions**: All experiment groups (e.g., `LSTM_SGD_MUP_SCALING`, `MUP_SCALING_EXPERIMENTS`, etc.)
- **Utility Functions**:
  - `subset_experiments()` - Filter experiments by label
  - `generate_lr_sweep_experiment()` - Create learning rate sweeps
  - `create_multi_lr_experiments()` - Create multiple learning rate variations
  - `create_multi_seed_experiments()` - Create multiple seed variations
- **Learning Rate Sweeps**: Predefined learning rate ranges (`STANDARD_LR_SWEEP`, `NARROW_LR_SWEEP`)
- **Default Configuration**: The `EXPERIMENTS` variable that defines which experiments to run by default

### `lstm_experiments.py`

The main experiment runner that:

- Imports experiment definitions from `lstm_experiment_definitions.py`
- Contains the base configuration (`CONFIG`)
- Handles distributed training setup
- Manages experiment execution and logging

## Usage

### Running Experiments

```bash
# Run with default experiments (LSTM_SGD_MUP_SCALING)
python lstm_experiments.py

# Run with SLURM job array
python lstm_experiments.py --job_id 0 --total_jobs 4
```

### Modifying Experiments

#### Option 1: Change the default in definitions file

Edit `lstm_experiment_definitions.py` and modify the `EXPERIMENTS` variable at the bottom:

```python
# Change default experiments
EXPERIMENTS = MUP_SCALING_EXPERIMENTS
```

#### Option 2: Override in the main file

Edit `lstm_experiments.py` and add an override after the imports:

```python
# Override default experiments
EXPERIMENTS = LSTM_LR_EXPERIMENTS
```

#### Option 3: Use subset selection

```python
# Run only specific experiments
wanted = {"lstm_16d_sgd_mup", "lstm_24d_sgd_mup"}
EXPERIMENTS = subset_experiments(LSTM_SGD_MUP_SCALING, wanted)
```

### Creating New Experiments

1. **Add to definitions file**: Add new experiment groups to `lstm_experiment_definitions.py`
2. **Use utility functions**: Leverage existing functions for common patterns:
   - Learning rate sweeps: `generate_lr_sweep_experiment()`
   - Multi-seed experiments: `create_multi_seed_experiments()`
   - Multi-learning rate experiments: `create_multi_lr_experiments()`

## Experiment Structure

Each experiment follows this structure:

```python
{
    "name": "experiment_group_name",
    "subexperiments": [
        {
            "label": "unique_experiment_label",
            "overrides": {
                "learning_rate": 1e-3,
                "hidden_size": 32,
                # ... other parameter overrides
            }
        }
    ]
}
```

## Benefits of This Structure

1. **Separation of Concerns**: Experiment definitions are separate from execution logic
2. **Reusability**: Utility functions can be used across different experiment types
3. **Maintainability**: Easy to add new experiments without modifying the main runner
4. **Flexibility**: Multiple ways to select and modify experiments
5. **Consistency**: Follows the same pattern as the transformer experiments in the parent directory



