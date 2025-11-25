# Compute Multiplier by Closest Approach

## Overview

The `compute_multiplier_closest_approach()` function compares two training curves (empirical or theoretical) to determine compute efficiency multipliers at their "closest approach" point.

**Closest Approach** is defined as the point where the two curves are closest together in terms of x-axis (compute) distance, not y-axis (loss) distance.

## What It Does

Given two curves:
1. Finds the point where they are closest in compute (x-axis distance)
2. At this closest point, computes the ratio of compute needed: `compute_B / compute_A`
3. Returns the multiplier showing how much more (or less) efficient curve A is compared to curve B

## Supported Comparison Types

### 1. CSV (Empirical) vs Power Law (Theoretical)

Compare an empirical training curve from a CSV file against a theoretical power law scaling law.

**Power law form:** `Loss = E + A * C^(-alpha)`

Where:
- `E` = Irreducible loss (asymptotic minimum)
- `A` = Amplitude/scale parameter
- `alpha` = Scaling exponent
- `C` = Compute in FLOPs

### 2. Power Law vs Power Law

Compare two theoretical power law scaling curves directly.

## Usage Examples

### Example 1: CSV vs Power Law (Legacy Syntax)

```python
from compute_multiplier_by_loss import compute_multiplier_closest_approach

# Power law parameters: Loss = E + A * C^(-alpha)
E = 3.5      # Irreducible loss
A = 1e14     # Amplitude
alpha = 0.05 # Exponent

csv_file = "path/to/training_curve.csv"

multiplier, details = compute_multiplier_closest_approach(
    csv_file,
    E=E,
    A=A,
    alpha=alpha,
    verbose=True
)

print(f"At closest approach, power law needs {multiplier:.3f}x compute vs CSV")
```

### Example 2: CSV vs Power Law (New Tuple Syntax)

```python
# Define power law as tuple: (E, A, alpha)
powerlaw_params = (3.5, 1e14, 0.05)

multiplier, details = compute_multiplier_closest_approach(
    "path/to/training_curve.csv",
    powerlaw_params,
    verbose=True
)
```

### Example 3: Power Law vs Power Law

```python
# Modern efficient transformer
powerlaw_modern = (3.5, 1e14, 0.05)  # (E, A, alpha)

# Older less efficient transformer  
powerlaw_old = (3.8, 2e14, 0.045)    # (E, A, alpha)

multiplier, details = compute_multiplier_closest_approach(
    powerlaw_modern,
    powerlaw_old,
    verbose=True,
    compute_range=(1e15, 1e18),  # Optional: specify compute range
    num_samples=2000             # Optional: number of samples
)

print(f"Old transformer needs {multiplier:.3f}x compute vs modern")
```

### Example 4: Power Law vs Power Law (Auto Range)

```python
# Let the function automatically determine the compute range
powerlaw_a = (4.0, 5e13, 0.06)
powerlaw_b = (4.2, 8e13, 0.055)

multiplier, details = compute_multiplier_closest_approach(
    powerlaw_a,
    powerlaw_b,
    verbose=True
    # compute_range is automatically determined
)
```

## Function Parameters

### Required Parameters

- **`input_a`**: First curve - either:
  - CSV file path (str): `"path/to/file.csv"`
  - Power law tuple: `(E, A, alpha)`

- **`input_b`**: Second curve (for comparison) - either:
  - Power law tuple: `(E, A, alpha)`
  - `None` (use legacy E, A, alpha parameters instead)

### Legacy Parameters (for backward compatibility)

- **`E`**: Irreducible loss (if `input_b=None`)
- **`A`**: Amplitude (if `input_b=None`)
- **`alpha`**: Exponent (if `input_b=None`)

### Optional Parameters

- **`compute_column`**: Name of compute column in CSV (default: `"total_flops_profiler"`)
- **`loss_column`**: Name of loss column in CSV (default: `"validation_loss"`)
- **`verbose`**: Print detailed information (default: `True`)
- **`compute_range`**: Tuple `(min_compute, max_compute)` for power law comparisons (default: auto-detect)
- **`num_samples`**: Number of samples for power law comparisons (default: `1000`)

## Return Values

Returns a tuple: `(multiplier, details)`

### multiplier (float)

The compute ratio at the closest approach point: `compute_B / compute_A`

**Interpretation:**
- `multiplier > 1`: Curve B needs MORE compute than curve A
- `multiplier < 1`: Curve B needs LESS compute than curve A
- `multiplier = 2.5`: Curve B needs 2.5x more compute than curve A

### details (dict)

Dictionary containing:

```python
{
    "comparison_type": "csv_vs_powerlaw" or "powerlaw_vs_powerlaw",
    "input_a": {...},  # Details about curve A
    "input_b": {...},  # Details about curve B
    "closest_approach": {
        "loss": float,           # Loss value at closest approach
        "compute_a": float,      # Compute for curve A (FLOPs)
        "compute_b": float,      # Compute for curve B (FLOPs)
        "x_distance": float,     # X-axis distance between curves
        "row_index": int,        # (CSV only) Row index in CSV
    },
    "multiplier": float,
    "compute_ratio_b_to_a": float,
}
```

## Algorithm Details

### CSV vs Power Law

1. For each point `(C_empirical, L_empirical)` on the empirical curve:
   - Solve for `C_powerlaw` where the power law reaches `L_empirical`:
     ```
     E + A * C_powerlaw^(-alpha) = L_empirical
     => C_powerlaw = (A / (L_empirical - E))^(1/alpha)
     ```
   - Compute x-axis distance: `|C_empirical - C_powerlaw|`

2. Find the point with minimum x-axis distance

3. Compute multiplier: `C_powerlaw / C_empirical` at this point

### Power Law vs Power Law

1. Sample compute values `C_a` across a range (log-spaced)

2. For each `C_a`:
   - Calculate loss on power law A: `L_a = E_a + A_a * C_a^(-alpha_a)`
   - Solve for `C_b` where power law B reaches `L_a`:
     ```
     E_b + A_b * C_b^(-alpha_b) = L_a
     => C_b = (A_b / (L_a - E_b))^(1/alpha_b)
     ```
   - Compute x-axis distance: `|C_a - C_b|`

3. Find the point with minimum x-axis distance

4. Compute multiplier: `C_b / C_a` at this point

## CSV File Requirements

CSV files must contain:
- A compute column (default: `total_flops_profiler`)
- A loss column (default: `validation_loss`)
- Numeric values (no NaN in rows to be compared)

Example CSV structure:
```csv
step,validation_loss,total_flops_profiler
0,10.5,1.0e15
1,9.2,2.0e15
2,8.1,3.0e15
...
```

## Interpretation Guide

### Example Results

**Result:** `multiplier = 2.5`
- Model B requires **2.5x more compute** than Model A to reach the same loss
- Model A is **2.5x more compute-efficient** than Model B
- Model B uses **60% more compute** than Model A (calculated as `(2.5-1)/2.5 * 100%`)

**Result:** `multiplier = 0.4`
- Model B requires **0.4x the compute** of Model A (60% less)
- Model A requires **2.5x more compute** than Model B (inverse: `1/0.4 = 2.5`)
- Model B is **2.5x more efficient** than Model A

## Notes and Limitations

1. **Irreducible Loss**: For power law comparisons, if the empirical loss is below or equal to `E`, that point is skipped (power law cannot reach it)

2. **Compute Range**: For power law vs power law, the automatic range detection may not always be optimal. You can manually specify `compute_range` if needed.

3. **Closest Approach**: The closest approach point may not always correspond to the most interesting or relevant comparison point for your analysis.

4. **Large Multipliers**: If you see very large multipliers (e.g., 1e200), this likely means:
   - The power law parameters are poorly fitted to the data
   - The two curves diverge significantly
   - The irreducible loss `E` is set too low

## Related Functions

- `compute_multiplier_by_loss()`: Compare compute at a specific loss value (not closest approach)
- `compare_multiple_loss_points()`: Compare across multiple loss values
- `analyze_loss_ranges()`: Analyze across the full overlapping loss range

## Questions or Issues?

For questions about this function or to report issues, please refer to the main project documentation or open an issue.

