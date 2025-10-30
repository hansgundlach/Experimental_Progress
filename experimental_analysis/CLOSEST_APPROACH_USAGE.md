# Using `compute_multiplier_closest_approach`

## Overview

The `compute_multiplier_closest_approach` function compares an empirical training curve to a power law scaling fit by finding the point of **closest approach** (minimum x-axis/compute distance) and computing the compute multiplier at that point.

## Function Signature

```python
compute_multiplier_closest_approach(
    csv_file: str,
    E: float,
    A: float,
    alpha: float,
    compute_column: str = 'total_flops_profiler',
    loss_column: str = 'validation_loss',
    verbose: bool = True
) -> Tuple[float, dict]
```

## How It Works

1. **Power Law Form**: `Loss = E + A * C^(-alpha)`
   - `E`: Irreducible loss (asymptotic minimum)
   - `A`: Amplitude (scaling constant)
   - `alpha`: Exponent (how fast loss decreases with compute)

2. **Closest Approach Algorithm**:
   - For each point `(C_empirical, L_empirical)` on the empirical curve:
     - Solve for `C_powerlaw` where the power law reaches the same loss: `C_powerlaw = (A / (L_empirical - E))^(1/alpha)`
     - Calculate x-axis distance: `|C_empirical - C_powerlaw|`
   - Find the point with **minimum distance** (closest approach)
   - Compute multiplier: `C_powerlaw / C_empirical` at that point

3. **Interpretation**:
   - `multiplier > 1`: Power law predicts MORE compute needed → empirical is MORE efficient
   - `multiplier < 1`: Power law predicts LESS compute needed → empirical is LESS efficient
   - `multiplier ≈ 1`: Both curves agree at this point

## Usage Example

```python
from experimental_analysis.main_ablation import compute_multiplier_closest_approach

# Step 1: Define your power law parameters (from fitting your scaling data)
E = 3.5      # Irreducible loss
A = 1e14     # Amplitude
alpha = 0.05 # Exponent

# Step 2: Path to your empirical CSV file
csv_file = "experimental_data_folder/stanford_mult/64d_rms.csv"

# Step 3: Compute the multiplier at closest approach
multiplier, details = compute_multiplier_closest_approach(
    csv_file=csv_file,
    E=E,
    A=A,
    alpha=alpha,
    compute_column="total_flops_profiler",
    loss_column="validation_loss",
    verbose=True
)

# Step 4: Interpret results
print(f"Multiplier: {multiplier:.3f}x")
print(f"Loss at closest approach: {details['closest_approach']['loss']:.4f}")
print(f"Empirical compute: {details['closest_approach']['empirical_compute']:.2e} FLOPs")
print(f"Power law compute: {details['closest_approach']['powerlaw_compute']:.2e} FLOPs")
```

## Use Cases

1. **Validate Scaling Laws**: Check if your empirical runs match theoretical predictions
2. **Identify Improvements**: Detect algorithmic improvements (empirical >> power law)
3. **Debug Training**: Find training issues (empirical << power law)
4. **Ablation Studies**: Compare modified architecture vs baseline scaling law

## Example Workflow

### 1. Fit Power Law to Baseline Scaling Data

First, collect loss vs compute data at different model sizes and fit:

```python
# Collect data from multiple model sizes
# Fit: Loss = E + A * C^(-alpha)
# Extract parameters E, A, alpha
```

### 2. Compare New Training Run

```python
# Run this function with your new empirical CSV
multiplier, details = compute_multiplier_closest_approach(
    csv_file="new_experiment.csv",
    E=fitted_E,
    A=fitted_A,
    alpha=fitted_alpha,
    verbose=True
)

# Multiplier > 1 means the new run is MORE efficient than baseline scaling
# Multiplier < 1 means the new run is LESS efficient than baseline scaling
```

### 3. Quantify Efficiency Gains

```python
if multiplier > 1:
    efficiency_gain = (multiplier - 1) * 100
    print(f"New method requires {efficiency_gain:.1f}% LESS compute!")
else:
    efficiency_loss = (1 - multiplier) * 100
    print(f"New method requires {efficiency_loss:.1f}% MORE compute")
```

## CSV File Requirements

Your CSV file must contain:

- A compute column (default: `total_flops_profiler`)
- A loss column (default: `validation_loss`)
- No NaN values in these columns

## Differences from `compute_multiplier_by_loss`

| Feature | `compute_multiplier_by_loss` | `compute_multiplier_closest_approach` |
|---------|----------------------------|-----------------------------------|
| **Compares** | Two empirical curves | Empirical curve vs power law fit |
| **At point** | Fixed target loss | Closest approach (min x-distance) |
| **Use case** | A/B testing | Validate against scaling law |
| **Inputs** | Two CSV files + target loss | One CSV + power law parameters |

## Tips

- Ensure your power law parameters are properly fitted to your baseline data
- Loss values in your CSV should be above `E` (irreducible loss)
- Use `verbose=True` for detailed output when debugging
- The closest approach point tells you where your empirical curve is most similar to the scaling law

## Import in main_ablation.py

The function is already imported in `main_ablation.py`:

```python
from experimental_analysis.main_ablation import compute_multiplier_closest_approach
```

You can also import it directly from the module:

```python
from experimental_analysis.compute_multiplier_by_loss import compute_multiplier_closest_approach
```

---

For questions or issues, refer to the function's docstring or the example usage in `main_ablation.py` (lines 512-541).
