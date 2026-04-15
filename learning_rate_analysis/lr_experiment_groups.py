"""
Learning rate experiment group definitions.

Each group defines a set of (hidden_dim, learning_rate) sweep experiments
sharing a common model configuration. After running, the analysis script
fits log(lr) = a + b*log(dim) and extrapolates optimal LRs for all sizes.

To add a new group, add an entry to LR_EXPERIMENT_GROUPS below.
"""

import numpy as np

# ============================================================================
# Common LR search grids (pick one or define your own per group)
# ============================================================================
WIDE_LR_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
STANDARD_LR_SWEEP = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
NARROW_LR_SWEEP = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SGD_LR_SWEEP = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]

# Standard dims to sweep (a few sizes to establish the LR-vs-dim relationship)
STANDARD_DIMS = [32, 64, 96, 128, 192, 256]
SMALL_DIMS = [32, 64, 128]

# ============================================================================
# Experiment group definitions
# ============================================================================
LR_EXPERIMENT_GROUPS = {
    "modern_transformer": {
        "description": "Modern transformer (SwiGLU, rotary, AdamW, cosine LR)",
        "hidden_dims": STANDARD_DIMS,
        "learning_rates": STANDARD_LR_SWEEP,
        "model_overrides": {
            "activation": "swiglu",
            "pos_encoding": "rotary",
            "optimizer": "adamw",
            "norm_type": "rms",
        },
        # Fraction of full Chinchilla token budget used for each LR trial
        "token_budget_fraction": 0.05,
        "csv_log_interval": 100,
    },
    "sgd_transformer": {
        "description": "Transformer with SGD + heavy-ball momentum",
        "hidden_dims": STANDARD_DIMS,
        "learning_rates": SGD_LR_SWEEP,
        "model_overrides": {
            "activation": "swiglu",
            "pos_encoding": "rotary",
            "optimizer": "sgd",
            "sgd_momentum": 0.98,
            "weight_decay": 0.0,
        },
        "token_budget_fraction": 0.05,
        "csv_log_interval": 100,
    },
    "historical_transformer": {
        "description": "2017-era transformer (sinusoidal pos, GELU, no RMSNorm)",
        "hidden_dims": STANDARD_DIMS,
        "learning_rates": STANDARD_LR_SWEEP,
        "model_overrides": {
            "activation": "gelu",
            "pos_encoding": "sinusoidal",
            "optimizer": "adamw",
            "norm_type": "layer",
        },
        "token_budget_fraction": 0.05,
        "csv_log_interval": 100,
    },
    # ---------------------------------------------------------------
    # Add your own groups below. Example:
    # ---------------------------------------------------------------
    # "my_new_group": {
    #     "description": "Description here",
    #     "hidden_dims": [32, 64, 128, 256],
    #     "learning_rates": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    #     "model_overrides": { ... },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
}


def list_groups():
    """Print all available experiment groups."""
    print("Available LR experiment groups:")
    print("-" * 60)
    for name, group in LR_EXPERIMENT_GROUPS.items():
        n_exps = len(group["hidden_dims"]) * len(group["learning_rates"])
        print(f"  {name:<30} {n_exps:>3} experiments")
        print(f"    {group['description']}")
        print(f"    dims: {group['hidden_dims']}")
        print(f"    LRs:  {[f'{lr:.0e}' for lr in group['learning_rates']]}")
    print("-" * 60)


if __name__ == "__main__":
    list_groups()
