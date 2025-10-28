# Scaling Law Analysis

This directory contains tools for fitting scaling laws to experimental data from transformer training experiments.

## Files

- `fit_hitchhikers_loss.py` - Main scaling law fitting implementation
- `run_scaling_analysis.py` - Helper script for easy dataset selection and analysis
- `README_scaling_analysis.md` - This documentation

## Quick Start

### Run All Available Datasets

```bash
python fit_hitchhikers_loss.py --all
```

### List Available Datasets

```bash
python run_scaling_analysis.py list
```

### Run Specific Dataset

```bash
python run_scaling_analysis.py optimal_lr_sgd_scaling
```

### Compare Multiple Datasets

```bash
python run_scaling_analysis.py optimal_lr_sgd_scaling vanilla_scaling_optimal_lr
```

## Available Datasets

The analysis automatically detects and configures the following dataset types:

1. **optimal_lr_sgd_scaling** - SGD experiments with optimal learning rates
2. **vanilla_scaling_optimal_lr** - Vanilla transformer scaling with optimal LR
3. **mup_scaling_experiments** - MuP (Maximal Update Parametrization) experiments
4. **vanilla_scaling** - Basic vanilla transformer scaling
5. **generated_experiments** - Generated experiment data

## Scaling Law Model

The code fits the following scaling law model:

```
L(N, D) = exp(E) + exp(A) * N^alpha + exp(B) * D^beta
```

Where:

- `L` = validation loss
- `N` = number of parameters
- `D` = number of training tokens
- `E, A, B` = log-space parameters
- `alpha, beta` = scaling exponents (typically negative)

## Parameter Counts

The analysis uses estimated parameter counts for different model sizes:

| Model Size | Parameters | Model Size | Parameters |
|------------|------------|------------|------------|
| 16d        | ~810k      | 64d        | ~6.1M      |
| 24d        | ~1.22M     | 80d        | ~9.5M      |
| 32d        | ~1.67M     | 96d        | ~13.7M     |
| 40d        | ~2.55M     | 128d       | ~24.4M     |
| 48d        | ~3.6M      |            |            |
| 56d        | ~4.8M      |            |            |

## Output Interpretation

### Key Metrics

- **Alpha (α)**: Parameter scaling exponent. More negative values indicate stronger parameter scaling.
- **Beta (β)**: Data scaling exponent. More negative values indicate stronger data scaling.
- **Exp(E)**: Irreducible loss - the theoretical minimum loss achievable.
- **Exp(A)**: Parameter scaling coefficient.
- **Exp(B)**: Data scaling coefficient.
- **Alpha*Beta/(Alpha+Beta)**: Combined scaling exponent.

### Example Results

```
DATASET: OPTIMAL_LR_SGD_SCALING
================================================================================
Fitted parameters (L = exp(E) + exp(A) * N^alpha + exp(B) * D^beta):
Log-space parameters:
  E     = 1.886821
  A     = -0.293439
  B     = 6.761058
Scaling exponents:
  alpha = -0.832533    # Strong parameter scaling
  beta  = -0.417971    # Moderate data scaling
Exponential form (actual scaling law coefficients):
  exp(E) = 6.598359  (irreducible loss)
  exp(A) = 0.745695  (parameter scaling coefficient)
  exp(B) = 863.555354  (data scaling coefficient)
  alpha*beta/(alpha+beta) = -0.278267
  num_points = 152
  final Huber loss = 0.021219
```

## Programmatic Usage

You can also use the fitting functions programmatically:

```python
from fit_hitchhikers_loss import fit_validation_loss_from_pairs

# Define your dataset
files_and_N = [
    ("path/to/32d_experiment.csv", 1666000),
    ("path/to/40d_experiment.csv", 2546000),
    ("path/to/64d_experiment.csv", 6100000),
]

# Fit the scaling law
result = fit_validation_loss_from_pairs(
    files_and_N,
    loss_column="validation_loss",
    use_tokens_column=True,
)

print(f"Alpha: {result.alpha}")
print(f"Beta: {result.beta}")
print(f"Irreducible loss: {result.exp_E}")
```

## Adding New Datasets

To add support for new datasets, modify the `get_dataset_configurations()` function in `fit_hitchhikers_loss.py`:

```python
def get_dataset_configurations() -> dict:
    # ... existing configurations ...
    
    # Add your new dataset
    new_dir = data_folder / "your_new_dataset"
    if new_dir.exists():
        new_pairs = [
            (new_dir / "file1.csv", param_counts[32]),
            (new_dir / "file2.csv", param_counts[40]),
            # ... more files
        ]
        configurations["your_new_dataset"] = [(str(p), n) for p, n in new_pairs if p.exists()]
    
    return configurations
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Pathlib (built-in)

## Notes

- The analysis uses Huber loss for robust fitting
- Two-phase optimization: Adam for exploration, LBFGS for refinement
- Results are printed in both log-space and exponential form for easier interpretation
- The code automatically handles missing files and provides informative error messages
