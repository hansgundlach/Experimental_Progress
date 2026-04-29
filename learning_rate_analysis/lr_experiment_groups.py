"""
Learning rate experiment group definitions.

Each group defines a set of (hidden_dim, learning_rate) sweep experiments
sharing a common model configuration. After running, the analysis script
fits log(lr) = a + b*log(dim) and extrapolates optimal LRs for all sizes.

To add a new group, add an entry to LR_EXPERIMENT_GROUPS below.

Two kinds of groups:
  - Concrete: defines `hidden_dims`, `learning_rates`, `model_overrides`,
    `token_budget_fraction`, `csv_log_interval`. Each (dim, lr) combo runs as
    one task and CSVs land in `results/<group>/`.
  - Combined: defines only `description` and `combine: [subgroup_a, ...]`.
    Aggregates CSVs from each listed concrete subgroup and fits a single
    LR scaling law across them. Useful when you want different
    `token_budget_fraction` (or other settings) per dim range but a unified
    fit. Combined groups are analyze-only — submit each subgroup separately,
    or pass the combined name to `submit_lr.sh` (which submits each sub
    array and an analyze job for the combined name).
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
    #  "modern_transformer_iter3": {
    #     "description": "Modern transformer (small dims, 5% Chinchilla budget)",
    #     "hidden_dims": [32, 48, 64, 96, 128, 256],
    #     "learning_rates": [10**-4.5, 10**-4.25, 10**-4,10**-3.75,10**-3.5,10**-3.25,10**-3,10**-2.75,10**-2.5,10**-2.25,10**-2,10**-1.75],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },

    # ========================================================================
    # FAIR-PROTOCOL SWEEPS (cross-arch comparable; iter2 measurements informed
    # the basin centers per arch).
    #
    # Design choices, applied identically to all three architectures:
    #   * Dim grid (transformers):        [32, 48, 64, 96, 128, 192, 256]
    #     Dim grid (LSTM):                [32, 48, 64, 96, 128, 192, 256, 384]
    #     (512 dropped: too costly given iter2 didn't need it for slope
    #     estimation; 96 / 48 / 192 added to densify the small-dim end where
    #     slope sensitivity is highest.)
    #   * Same LR-grid resolution:        0.25 decades
    #   * Same token budget per trial:    5% Chinchilla
    #   * Small/large split per arch:     basin sits at higher LR for small
    #                                     dims, lower LR for large dims --
    #                                     a single grid wastes compute at
    #                                     the extremes. Split mirrors Bjorck
    #                                     et al. 2025 (ICLR) adaptive-grid
    #                                     refinement protocol.
    #   * >=7 LRs per dim within the basin so the parabolic vertex
    #     (Bjorck 2025; VaultGemma 2025) has enough signal; for historical
    #     the large group adds 0.1-decade infill across the post-LN cliff
    #     (Vaswani 2017; Xiong et al. 2020 -- "On Layer Normalization in the
    #     Transformer Architecture", ICML 2020).
    #
    # Per-arch basin centers (log10 η*) measured from iter2:
    #   modern transformer:   d=32 -> -1.97 ;  d=512 ext. -> -3.27
    #   historical transformer (post-LN cliff):
    #                          d=32 -> -2.25 ;  d>=128 cliff at -2.75
    #   LSTM (1- and 2-layer): d=32 -> -1.45 ;  d=512 -> -2.18
    # ========================================================================

    # ---------- Modern transformer (smooth basin) ----------
    # "modern_transformer_fair_small": {
    #     "description": "Modern transformer (small dims, fair-protocol)",
    #     "hidden_dims": [32, 48, 64, 96],
    #     # 9 LRs at 0.25-dec spacing covering the small-dim basin.
    #     "learning_rates": [
    #         10**-3.25, 10**-3.0, 10**-2.75, 10**-2.5, 10**-2.25,
    #         10**-2.0, 10**-1.75, 10**-1.5, 10**-1.25,
    #     ],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "modern_transformer_fair_large": {
    #     "description": "Modern transformer (large dims, fair-protocol)",
    #     "hidden_dims": [128, 192, 256],
    #     # 9 LRs at 0.25-dec spacing covering the large-dim basin; range
    #     # extended upward to -2.0 to avoid the d=128 boundary-vertex bug
    #     # we saw in iter2.
    #     "learning_rates": [
    #         10**-4.0, 10**-3.75, 10**-3.5, 10**-3.25, 10**-3.0,
    #         10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0,
    #     ],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "modern_transformer_fair_combined": {
    #     "description": "Combined fit for fair-protocol modern transformer",
    #     "combine": ["modern_transformer_fair_small", "modern_transformer_fair_large"],
    # },

    # ---------- Historical transformer (post-LN cliff) ----------
    # Same grid as modern for the small dims; large dims have 0.1-dec infill
    # across the cliff zone so the analyzer can localize where the model
    # diverges instead of just hitting the plateau.
    "historical_transformer_fair_small_iter4": {
        "description": "Historical (post-LN, sinusoidal, inv-sqrt) small dims, fair-protocol",
        "hidden_dims": [32, 48, 64, 80, 96],
        "learning_rates": [
            10**-3.25, 10**-3.0, 10**-2.75, 10**-2.5, 10**-2.25,
            10**-2.0, 10**-1.75, 10**-1.5, 10**-1.25,
        ],
        "model_overrides": {
            "activation": "gelu",
            "pos_encoding": "sinusoidal",
            "norm_placement": "post",
            "lr_schedule": "inverse_sqrt",
            "optimizer": "adam",
            "norm_type": "layer",
            "modern_bias_0": False,
            "weight_decay": 0.01,
            "dropout": 0.0,
            "modern_bias_0": False,
            "ff_ratio": 4,
        },
        "token_budget_fraction": 1.0,
        "csv_log_interval": 500,
    },
    # "historical_transformer_fair_large_iter4": {
    #     "description": "Historical (post-LN) large dims with cliff-zone infill, fair-protocol",
    #     "hidden_dims": [128, 192, 256],
    #     # 0.25-dec backbone PLUS 0.1-dec infill at [-3.0, -2.9, -2.8, -2.7]
    #     # to bracket the post-LN divergence cliff observed in iter2 between
    #     # log10(lr) -3.0 and -2.75.
    #     "learning_rates": [
    #         10**-4.25, 10**-4.0, 10**-3.75, 10**-3.5, 10**-3.25, 10**-3.0,
    #         10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0,
    #     ],
    #     "model_overrides": {
    #         "activation": "gelu",
    #         "pos_encoding": "sinusoidal",
    #         "norm_placement": "post",
    #         "lr_schedule": "inverse_sqrt",
    #         "optimizer": "adam",
    #         "norm_type": "layer",
    #         "modern_bias_0": False,
    #         "weight_decay": 0.01,
    #         "dropout": 0.0,
    #         "modern_bias_0": False,
    #         "ff_ratio": 4,
    #     },
    #     "token_budget_fraction": 0.2,
    #     "csv_log_interval": 500,
    # },
    # "historical_transformer_fair_combined_iter4": {
    #     "description": "Combined fit for fair-protocol historical transformer",
    #     "combine": ["historical_transformer_fair_small_iter4", "historical_transformer_fair_large_iter4"],
    # },

    # ---------- LSTM (smooth basin, single sweep grid suffices) ----------
    # LSTM optimum sits ~1 decade above transformer optimum; basin is shallow
    # enough that a single grid covers all dims. 8 LRs at 0.25-dec spacing.
    "lstm_fair_layer2": {
        "description": "LSTM 2-layer (fair-protocol, single grid covers all dims)",
        "architecture": "lstm",
        "hidden_dims": [32, 48, 64, 96, 128],
        "learning_rates": [
            10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0,
            10**-1.75, 10**-1.5, 10**-1.25, 10**-1.0,
        ],
        "model_overrides": {
            "num_layers": 2,
        },
        "token_budget_fraction": 1.0,
        "csv_log_interval": 100,
    },
    "lstm_fair_layer1": {
        "description": "LSTM 1-layer (fair-protocol, single grid covers all dims)",
        "architecture": "lstm",
        "hidden_dims": [32, 48, 64, 96, 128],
        "learning_rates": [
            10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0,
            10**-1.75, 10**-1.5, 10**-1.25, 10**-1.0,
        ],
        "model_overrides": {
            "num_layers": 1,
        },
        "token_budget_fraction": 1.0,
        "csv_log_interval": 100,
    },
    # "historical_transformer_iter3": {
    #     "description": "Historical transformer (sinusoidal pos, GELU, no RMSNorm)",
    #     "hidden_dims": [32, 48, 64, 96, 128, 256],
    #     "learning_rates": [10**-4.5, 10**-4.25, 10**-4,10**-3.75,10**-3.5,10**-3.25,10**-3,10**-2.75,10**-2.5,10**-2.25,10**-2,10**-1.75],
    #     "model_overrides": {
    #         "activation": "gelu",
    #         "pos_encoding": "sinusoidal",
    #         "norm_placement": "post",
    #         "lr_schedule": "inverse_sqrt",
    #         "optimizer": "adam",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },






















    # "modern_transformer_iter2": {
    #     "description": "Modern transformer (SwiGLU, rotary, AdamW, cosine LR)",
    #     "hidden_dims": [32, 64, 128, 192, 256, 512],
    #     "learning_rates": [10**-4.5, 10**-4.25, 10**-4,10**-3.75,10**-3.5,10**-3.25,10**-3,10**-2.75,10**-2.5,10**-2.25,10**-2,10**-1.75],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     # Fraction of full Chinchilla token budget used for each LR trial
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "sgd_transformer": {
    #     "description": "Transformer with SGD + heavy-ball momentum",
    #     "hidden_dims": STANDARD_DIMS,
    #     "learning_rates": SGD_LR_SWEEP,
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "sgd",
    #         "sgd_momentum": 0.98,
    #         "weight_decay": 0.0,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "historical_transformer_iter2": {
    #     "description": "2017-era transformer (sinusoidal pos, GELU, no RMSNorm)",
    #     "hidden_dims": [32, 64, 128, 192, 256, 512],
    #     "learning_rates": [10**-4.5, 10**-4.25, 10**-4,10**-3.75,10**-3.5,10**-3.25,10**-3,10**-2.75,10**-2.5,10**-2.25,10**-2,10**-1.75],
    #     "model_overrides": {
    #         "activation": "gelu",
    #         "pos_encoding": "sinusoidal",
    #         "norm_placement": "post",
    #         "lr_schedule": "inverse_sqrt",
    #         "optimizer": "adam",
    #         "norm_type": "layer",
    #         "modern_bias_0": False,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "historical_transformer_iter2_small": {
    #     "description": "2017-era transformer (small dims, 5% Chinchilla budget)",
    #     "hidden_dims": [32, 64],
    #     "learning_rates": [10**-3, 10**-2.75, 10**-2.5, 10**-2.25, 10**-2, 10**-1.75],
    #     "model_overrides": {
    #         "activation": "gelu",
    #         "pos_encoding": "sinusoidal",
    #         "norm_placement": "post",
    #         "lr_schedule": "inverse_sqrt",
    #         "optimizer": "adam",
    #         "norm_type": "layer",
    #         "modern_bias_0": False,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "historical_transformer_iter2_large": {
    #     "description": "2017-era transformer (large dims, 4% Chinchilla budget)",
    #     "hidden_dims": [128, 256, 320],
    #     "learning_rates": [10**-4.5, 10**-4.25, 10**-4, 10**-3.75, 10**-3.5, 10**-3.25, 10**-3, 10**-2.75, 10**-2.5, 10**-2.25],
    #     "model_overrides": {
    #         "activation": "gelu",
    #         "pos_encoding": "sinusoidal",
    #         "norm_placement": "post",
    #         "lr_schedule": "inverse_sqrt",
    #         "optimizer": "adam",
    #         "norm_type": "layer",
    #         "modern_bias_0": False,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "historical_transformer_iter2_combined": {
    #     "description": "Combined small+large fit for 2017-era transformer iter2",
    #     "combine": [
    #         "historical_transformer_iter2_small",
    #         "historical_transformer_iter2_large",
    #     ],
    # },



    # "lstm_layer2_large": {
    #     "description": "LSTM with AdamW optimizer and cosine LR schedule",
    #     "architecture": "lstm",
    #     "hidden_dims": [32, 64, 128, 256, 384, 512],
    #     "learning_rates": [10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1],
    #     "model_overrides": {
    #         "num_layers": 2,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "lstm_layer1_large": {
    #     "description": "LSTM with AdamW optimizer and cosine LR schedule",
    #     "architecture": "lstm",
    #     "hidden_dims": [32, 64, 128, 256, 384, 512],
    #     "learning_rates": [10**-2.75, 10**-2.5, 10**-2.25, 10**-2.0, 10**-1.75, 10**-1.5, 10**-1.25, 10**-1],
    #     "model_overrides": {
    #         "num_layers": 1,
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },


    # "modern_transformer_iter3_small": {
    #     "description": "Modern transformer (small dims, 5% Chinchilla budget)",
    #     "hidden_dims": [32, 48, 64, 96, 128, 256],
    #     "learning_rates": [10**-3.25, 10**-3, 10**-2.75, 10**-2.5, 10**-2.25, 10**-2, 10**-1.75],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "modern_transformer_iter3_large": {
    #     "description": "Modern transformer (large dims, 4% Chinchilla budget)",
    #     "hidden_dims": [128, 256],
    #     "learning_rates": [10**-3.75, 10**-3.5, 10**-3.25, 10**-3, 10**-2.75, 10**-2.5, 10**-2.25, 10**-2],
    #     "model_overrides": {
    #         "activation": "swiglu",
    #         "pos_encoding": "rotary",
    #         "optimizer": "adamw",
    #         "norm_type": "rms",
    #     },
    #     "token_budget_fraction": 0.05,
    #     "csv_log_interval": 100,
    # },
    # "modern_transformer_iter3_combined": {
    #     "description": "Combined small+large fit for modern transformer iter2",
    #     "combine": ["modern_transformer_iter2_small", "modern_transformer_iter2_large"],
    # },
}


# old config 
#  "modern_transformer": {                                                                 
#       "description": "Modern transformer (SwiGLU, rotary, AdamW, cosine LR)",                                                                                                                                                                                
#       "hidden_dims": STANDARD_DIMS,                  # [32, 64, 96, 128, 192, 256]                                                                                                                                                                           
#       "learning_rates": STANDARD_LR_SWEEP,           # [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]                                                                                                                                                            
#       "model_overrides": {                                                                                                                                                                                                                                   
#           "activation": "swiglu",                                                                                                                                                                                                                            
#           "pos_encoding": "rotary",                                                                                                                                                                                                                          
#           "optimizer": "adamw",                                                                                                                                                                                                                              
#           "norm_type": "rms",
#       },                                                                                                                                                                                                                         
#       "token_budget_fraction": 0.05,                     
#       "csv_log_interval": 100,
#   },    









def is_combined_group(group_name):
    """True if the group is a meta-entry that aggregates other groups."""
    g = LR_EXPERIMENT_GROUPS.get(group_name)
    return bool(g and "combine" in g)


def expand_subgroups(group_name):
    """
    Return the list of concrete subgroup names backing this group.

    Concrete group  -> [group_name]
    Combined group  -> its `combine` list (validated to exist and be concrete)
    """
    if group_name not in LR_EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown LR group: '{group_name}'")
    g = LR_EXPERIMENT_GROUPS[group_name]
    if "combine" not in g:
        return [group_name]
    subs = g["combine"]
    if not isinstance(subs, list) or not subs:
        raise ValueError(f"'{group_name}'.combine must be a non-empty list")
    for s in subs:
        if s not in LR_EXPERIMENT_GROUPS:
            raise ValueError(f"'{group_name}' references unknown subgroup '{s}'")
        if "combine" in LR_EXPERIMENT_GROUPS[s]:
            raise ValueError(f"'{group_name}' references combined subgroup '{s}' (nesting not supported)")
    return list(subs)


def list_groups():
    """Print all available experiment groups."""
    print("Available LR experiment groups:")
    print("-" * 60)
    for name, group in LR_EXPERIMENT_GROUPS.items():
        if "combine" in group:
            print(f"  {name:<30} [combined]")
            print(f"    {group['description']}")
            print(f"    subgroups: {group['combine']}")
        else:
            n_exps = len(group["hidden_dims"]) * len(group["learning_rates"])
            print(f"  {name:<30} {n_exps:>3} experiments")
            print(f"    {group['description']}")
            print(f"    dims: {group['hidden_dims']}")
            print(f"    LRs:  {[f'{lr:.0e}' for lr in group['learning_rates']]}")
    print("-" * 60)


if __name__ == "__main__":
    list_groups()
