import torch
import torch.distributed as dist
import os
import time
import copy
import math
import sys

# Ensure local directory is on sys.path for relative imports when launched by Slurm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lstm_training import train_model
import argparse
import torch.multiprocessing as mp
from socket import socket
from typing import List, Union, Sequence

# Configuration
# has 1.66 total params
CONFIG = {
    "data_path": "../Datasets/c4_subset.txt",
    "tokenizer_path": "../gpt2_tokenizer",
    "max_characters": 5 * 1e7,  # Maximum number of characters to use from dataset
    "sequence_length": 128,
    "batch_size": 32,  # Keep physical batch size small, has no effect on model
    "hidden_size": 16,
    "num_layers": 2,
    "dropout": 0.0,  # dropout zer here to match transformer but may need to adjust for LSTM
    "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
    "lr_schedule": "cosine",
    "step_size": 10,
    "gamma": 0.1,  # parameter usedf for stepLR step decay
    "num_epochs": 1,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb_project": "lstm-wikitext",
    "wandb_offline": True,
    "print_every": 100,  # Print loss every N batches
    # Gradient clipping settings
    "use_gradient_clipping": True,
    "gradient_clip_val": 1.0,
    # NEW: CSV logging settings
    "results_folder": "Experiments_Folder",
    "csv_log_interval": 50,  # Log every 100 steps
    # NEW: Data loading optimization settings
    "num_workers": "auto",  # Will be set automatically based on CPU cores
    "pin_memory": True,  # Faster GPU memory transfer
    "persistent_workers": True,  # Keep data loading workers alive between epochs
    "prefetch_factor": 4,  # Number of batches to prefetch per worker
    # NEW: Mixed precision settings
    "use_amp": False,  # Enable Automatic Mixed Precision
    "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
    # NEW: Gradient accumulation settings
    "gradient_accumulation_steps": 16,  # For tracking only
    # NEW: whether to compile the model (PyTorch 2.0+)
    "use_compile": False,
    "seed": 123,
    "optimizer": "adamw",  # NEW: choose from "adam", "adamw", or "sgd"
    "weight_decay": 0.01,
    "stride": 64,  # NEW: sliding-window stride to match transformer
    # Add three separate variational dropout parameters
    "input_dropout": 0.2,  # Applied to embeddings
    "hidden_dropout": 0.1,  # Applied between LSTM layers
    "output_dropout": 0.2,  # Applied before final linear layer
    "use_layer_norm": True,  # Enable/disable LayerNorm
    "layer_norm_position": "output",  # Options: "input", "output", "both", "gates"
}

# old large 5-6M param config:
# CONFIG = {
#     "data_path": "../Datasets/wikitext.txt",
#     "tokenizer_path": "../gpt2_tokenizer",
#     "max_characters": 3 * 1e8,  # Maximum number of characters to use from dataset
#     "sequence_length": 128,
#     "batch_size": 32,  # Keep physical batch size small
#     "hidden_size": 64,
#     "num_layers": 2,
#     "dropout": 0.2,
#     "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
#     "lr_schedule": "cosine",
#     "step_size": 10,
#     "gamma": 0.1,
#     "num_epochs": 4,
#     "train_split": 0.8,
#     "val_split": 0.1,
#     "test_split": 0.1,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "wandb_project": "lstm-wikitext",
#     "wandb_offline": True,
#     "print_every": 100,  # Print loss every N batches
#     # Gradient clipping settings
#     "use_gradient_clipping": True,
#     "gradient_clip_val": 1.0,
#     # NEW: Data loading optimization settings
#     "num_workers": "auto",  # Will be set automatically based on CPU cores
#     "pin_memory": True,  # Faster GPU memory transfer
#     "persistent_workers": True,  # Keep data loading workers alive between epochs
#     "prefetch_factor": 4,  # Number of batches to prefetch per worker
#     # NEW: Mixed precision settings
#     "use_amp": True,  # Enable Automatic Mixed Precision
#     "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
#     # NEW: Gradient accumulation settings
#     "gradient_accumulation_steps": 4,  # Simulate 4x larger batch size (32*4 = 128)
#     "effective_batch_size": 128,  # For tracking only - computed from batch_size * gradient_accumulation_steps
# }


# ========= Experiment definitions (customize labels & overrides below) =========
TEST_EXPERIMENTS = [
    {
        "name": "LSTM_benchmark",
        "subexperiments": [
            {
                "label": "LSTM_1.6M_Benchmark",
                "overrides": {"learning_rate": 0.001 * math.sqrt(4), "hidden_size": 16},
            },
        ],
    },
]


LSTM_OPTIMIZER_EXPERIMENTS = [
    {
        "name": "LSTM_Optimizer_Experiments",
        "subexperiments": [
            {
                "label": "LSTM_adam",
                "overrides": {"optimizer": "adam"},
            },
            {
                "label": "LSTM_SGD_Benchmark",
                "overrides": {"optimizer": "sgd"},
            },
        ],
    },
]


LSTM_OPTIMAL_SCALING = [
    {
        "name": "lstm_optimal_scaling",
        "subexperiments": [
            {
                "label": "lstm_16d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 258e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 388e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 519e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
        ],
    },
]


# – Add more experiments here, e.g.
# {
#   "name": "Another_experiment",
#   "subexperiments": [
#     { "label": "foo", "overrides": {...} },
#     …
#   ]
# },
# ]

# ===
# 193.7e6 258.6e6 388.8e6 520e6


LSTM_HIDDEN_DIM_EXPERIMENTS_LR_TUNES = [
    {
        "name": "lstm_hidden_dim_scaling_lr_tunes",
        "subexperiments": [
            {
                "label": "lstm_16d_1e-3",
                "overrides": {
                    "learning_rate": 1e-3,
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d_1e-3",
                "overrides": {
                    "learning_rate": 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d_1e-3",
                "overrides": {
                    "learning_rate": 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d_1e-3",
                "overrides": {
                    "learning_rate": 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d_1e-3",
                "overrides": {
                    "learning_rate": 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
            {
                "label": "lstm_16d_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
            {
                "label": "lstm_16d_1e-2",
                "overrides": {
                    "learning_rate": 1e-2,
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d_1e-2",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d_1e-2",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d_1e-2",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d_1e-2",
                "overrides": {
                    "learning_rate": 1e-2,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
            {
                "label": "lstm_16d_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
            {
                "label": "lstm_16d_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "lstm_24d_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_32d_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "lstm_48d_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
            {
                "label": "LSTM_64d_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                },
            },
        ],
    },
]


LSTM_SGD_SCALING = [
    {
        "name": "lsmt_sgd_scaling",
        "subexperiments": [
            {
                "label": "lstm_16d_sgd",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "lstm_16d_sgd_mup",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                    "optimizer": "sgd",
                    "use_mup": True,
                },
            },
            {
                "label": "LSTM_24d_sgd",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "LSTM_32d_sgd",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 32,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "LSTM_48d_sgd",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 48,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "LSTM_64d_sgd",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                    "optimizer": "sgd",
                },
            },
            {
                "label": "LSTM_64d_sgd_mup",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 129e6,
                    "seed": 123,
                    "hidden_size": 64,
                    "optimizer": "sgd",
                    "use_mup": True,
                },
            },
        ],
    },
]

# lsmt variaionts where originally done with 10^-3


LSTM_VARIATIONS = [
    {
        "name": "lstm_variations",
        "subexperiments": [
            {
                "label": "lstm_24d_layernorm",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "lstm_24d_3_layers",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                    "num_layers": 3,
                },
            },
            {
                "label": "lstm_24d_1_layers",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                    "num_layers": 1,
                },
            },
            {
                "label": "lstm_24d_no_layer_norm",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": False,
                },
            },
            {
                "label": "lstm_24d_cosine_warmup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "lr_schedule": "cosine_warmup",
                },
            },
            {
                "label": "lstm_24d_inverse_sqrt",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "lr_schedule": "inverse_sqrt",
                },
            },
        ],
    },
]

LSTM_LR_EXPERIMENTS = [
    {
        "name": "lstm_lr_experiments",
        "subexperiments": [
            {
                "label": "24d_lstm_lr_1e-1",
                "overrides": {
                    "learning_rate": 10 ** (-1),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-1.5",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-2",
                "overrides": {
                    "learning_rate": 10 ** (-2),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "stride": 128,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-2.5",
                "overrides": {
                    "learning_rate": 10 ** (-2.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-3",
                "overrides": {
                    "learning_rate": 10 ** (-3),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-3.5",
                "overrides": {
                    "learning_rate": 10 ** (-3.5),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
            {
                "label": "24d_lstm_lr_1e-4",
                "overrides": {
                    "learning_rate": 10 ** (-4),
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": True,
                },
            },
        ],
    },
]


# {
#                 "label": "LSTM_64d_123",
#                 "overrides": {
#                     "learning_rate": 1e-3,
#                     "max_characters": 519.9e6,
#                     "seed": 123,
#                     "hidden_size": 64,
#                 },
#             },


# EXPERIMENTS = [
#     {
#         "name": "Diag_experiment_1",
#         "subexperiments": [
#             {
#                 "label": "My label for sub experiment 1",
#                 "overrides": {"learning_rate": 1e-2},
#             },
#             {
#                 "label": "My label for sub experiment 2",
#                 "overrides": {"sequence_length": 64},
#             },
#             {
#                 "label": "My label for sub experiment 3",
#                 "overrides": {"lr_schedule": "cosine", "step_size": 20},
#             },
#         ],
#     },
#     # – Add more experiments here, e.g.
#     # {
#     #   "name": "Another_experiment",
#     #   "subexperiments": [
#     #     { "label": "foo", "overrides": {...} },
#     #     …
#     #   ]
#     # },
# ]

MUP_SCALING_EXPERIMENTS = [
    {
        "name": "muP_scaling_experiments",
        "subexperiments": [
            {
                "label": "lstm_16d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 24,  # Base width for muP scaling
                },
            },
            {
                "label": "lstm_24d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),
                    "hidden_size": 24,
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 24,  # Base width for muP scaling
                },
            },
            {
                "label": "lstm_32d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR as 16d
                    "hidden_size": 32,
                    "max_characters": 258.6e6,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 24,  # Same base width - muP should handle scaling
                },
            },
            {
                "label": "lstm_64d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR - muP handles scaling
                    "hidden_size": 64,
                    "max_characters": 519.9e6,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 24,  # Same base width
                },
            },
            {
                "label": "lstm_128d_mup",
                "overrides": {
                    "learning_rate": 10 ** (-1.5),  # Same base LR - muP handles scaling
                    "hidden_size": 128,
                    "max_characters": 1050e6,
                    "seed": 123,
                    "use_mup": True,
                    "mup_base_width": 24,  # Same base width
                },
            },
        ],
    },
]


def subset_experiments(experiment_list, wanted_labels):
    """Return only the sub-experiments whose 'label' is in wanted_labels,
    keeping each group's original name unchanged."""
    result = []
    for exp in experiment_list:
        picked = [se for se in exp["subexperiments"] if se["label"] in wanted_labels]
        if picked:
            # keep exp['name'] exactly as-is
            result.append(
                {
                    "name": exp["name"],
                    "subexperiments": picked,
                }
            )
    return result


def generate_lr_sweep_experiment(base_label, learning_rates, base_overrides=None):
    """
    Generate a learning rate sweep experiment based on a base configuration.

    Args:
        base_label: Base label for the experiment (e.g., "lstm_32d_mup")
        learning_rates: List of learning rate values to sweep over
        base_overrides: Dictionary of base configuration overrides (optional)

    Returns:
        Dictionary containing the experiment with all learning rate variations
    """
    if base_overrides is None:
        base_overrides = {}

    subexperiments = []

    for lr in learning_rates:
        # Create label with learning rate suffix
        # Format learning rate for filename-safe label
        if lr >= 1:
            lr_str = f"{lr:.0f}"
        elif lr >= 0.01:
            lr_str = f"{lr:.3f}".rstrip("0").rstrip(".")
        else:
            # For very small learning rates, use scientific notation
            lr_str = f"{lr:.1e}".replace("-", "m").replace("+", "p")

        label = f"{base_label}_lr_{lr_str}"

        # Create overrides with the learning rate
        overrides = copy.deepcopy(base_overrides)
        overrides["learning_rate"] = lr

        subexperiments.append({"label": label, "overrides": overrides})

    return {"name": f"{base_label}_lr_sweep", "subexperiments": subexperiments}


def create_multi_lr_experiments(base_experiments, learning_rates):
    """
    Create multiple versions of experiments with different learning rates.
    Similar to create_multi_seed_experiments but for learning rates.

    Args:
        base_experiments: List of experiment dictionaries (e.g., LSTM_HIDDEN_DIM_EXPERIMENTS)
        learning_rates: List of learning rate values (e.g., [1e-4, 1e-3, 1e-2])

    Returns:
        List of experiment dictionaries with learning rate variations
    """
    multi_lr_experiments = []

    for experiment in base_experiments:
        # Create a new experiment group for each base experiment
        new_experiment = {
            "name": f"{experiment['name']}_lr_sweep",
            "subexperiments": [],
        }

        # For each subexperiment in the base experiment
        for sub_exp in experiment["subexperiments"]:
            # Create a version for each learning rate
            for lr in learning_rates:
                # Create new subexperiment with lr suffix
                new_sub_exp = copy.deepcopy(sub_exp)

                # Add learning rate to the label
                original_label = sub_exp["label"]
                # Format learning rate for filename-safe label
                if lr >= 1:
                    lr_str = f"{lr:.0f}"
                elif lr >= 0.01:
                    lr_str = f"{lr:.3f}".rstrip("0").rstrip(".")
                else:
                    # For very small learning rates, use scientific notation
                    lr_str = f"{lr:.1e}".replace("-", "m").replace("+", "p")

                new_sub_exp["label"] = f"{original_label}_lr_{lr_str}"

                # Add learning rate to overrides
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["learning_rate"] = lr
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                else:
                    # If neither exists, create overrides with just the learning rate
                    new_sub_exp["overrides"] = {"learning_rate": lr}

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    return multi_lr_experiments


# ====================================================================
# LEARNING RATE SWEEP EXAMPLES AND DEFINITIONS
# ====================================================================

# Define standard learning rate sweeps
# STANDARD_LR_SWEEP = [1e-4, 10 ** (-3.5), 1e-3, 10 ** (-2.5), 1e-2, 10 ** (-1.5), 1e-1]
NARROW_LR_SWEEP = [
    10 ** (-3),
    10 ** (-2.5),
    10 ** (-2),
    10 ** (-1.5),
    1e-1,
]  # Focused sweep around promising values

subset_experiments = subset_experiments(
    LSTM_SGD_SCALING, {"lstm_16d_sgd_mup", "lstm_16_sgd"}
)
# # Active configuration: Learning rate sweep on lstm_32d

EXPERIMENTS = create_multi_lr_experiments(subset_experiments, NARROW_LR_SWEEP)

# EXPERIMENTS = MUP_SCALING_EXPERIMENTS
# Example usage patterns:

# Example 1: Generate a single learning rate sweep experiment
# EXPERIMENTS = [
#     generate_lr_sweep_experiment(
#         "lstm_32d_mup",
#         NARROW_LR_SWEEP,
#         base_overrides={
#             "hidden_size": 32,
#             "max_characters": 258.6e6,
#             "seed": 123,
#             "use_mup": True,
#             "mup_base_width": 24,
#         }
#     )
# ]

# Example 2: Apply learning rate sweep to existing experiment groups
# EXPERIMENTS = create_multi_lr_experiments(LSTM_HIDDEN_DIM_EXPERIMENTS, STANDARD_LR_SWEEP)

# Example 3: Combine with subset selection for targeted sweeps
# wanted = {"lstm_32d"}
# selected_experiments = subset_experiments(LSTM_HIDDEN_DIM_EXPERIMENTS, wanted)
# EXPERIMENTS = create_multi_lr_experiments(selected_experiments, NARROW_LR_SWEEP)

# Example 4: Multiple learning rate sweeps for different base configurations
# EXPERIMENTS = [
#     generate_lr_sweep_experiment("lstm_24d_adamw", STANDARD_LR_SWEEP, {"hidden_size": 24, "optimizer": "adamw"}),
#     generate_lr_sweep_experiment("lstm_24d_sgd", STANDARD_LR_SWEEP, {"hidden_size": 24, "optimizer": "sgd"}),
# ]


# ============================================================================
# SELECT WHICH EXPERIMENTS TO RUN
# ============================================================================

# Option 1: Use predefined experiment sets
# EXPERIMENTS = MUP_SCALING_EXPERIMENTS
# EXPERIMENTS = LSTM_HIDDEN_DIM_EXPERIMENTS
# EXPERIMENTS = LSTM_LR_EXPERIMENTS

# Option 2: Use subset of experiments
# wanted = {"LSTM_128d_123"}
# EXPERIMENTS = subset_experiments(MUP_SCALING_EXPERIMENTS, {"lstm_128d_mup"})

# Option 3: Learning rate sweeps on single configuration
# EXPERIMENTS = [
#     generate_lr_sweep_experiment(
#         "lstm_24d_mup",
#         STANDARD_LR_SWEEP,
#         base_overrides={
#             "hidden_size": 24,
#             "max_characters": 193.7e6,
#             "seed": 123,
#             "use_mup": True,
#             "mup_base_width": 24,
#         }
#     )
# ]

# Option 4: Learning rate sweeps on multiple experiment groups
# EXPERIMENTS = create_multi_lr_experiments(LSTM_HIDDEN_DIM_EXPERIMENTS, NARROW_LR_SWEEP)

# Default: Use original MUP scaling experiments
# EXPERIMENTS = MUP_SCALING_EXPERIMENTS


def find_free_port():
    """Finds a free port on the host."""
    with socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_ddp_worker(
    local_rank, world_size, master_addr, master_port, config, run_name, csv_log_path
):
    """
    This function is spawned for each DDP process.
    """
    # 1. --- Set up DDP environment ---
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    # Update config with device for this rank
    config["device"] = f"cuda:{local_rank}"

    if local_rank == 0:
        print(f"\n--- Starting DDP run for: {run_name} ---")
        print(f"  - World Size: {world_size}")
        print(f"  - Master: {master_addr}:{master_port}")

    # 2. --- Run the actual training ---
    # The csv_log_path is passed, but only rank 0 will write to it.
    train_model(
        config=config,
        local_rank=local_rank,
        run_name=run_name,
        csv_log_path=csv_log_path,
    )

    # 3. --- Clean up ---
    dist.destroy_process_group()

    if local_rank == 0:
        print(f"--- Completed DDP run for: {run_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM Experiments with DDP")
    parser.add_argument("--job_id", type=int, default=0, help="SLURM job array ID")
    parser.add_argument(
        "--total_jobs", type=int, default=1, help="Total SLURM jobs in array"
    )
    args = parser.parse_args()

    # Fallback: derive array info from SLURM env if not provided via CLI
    if args.total_jobs == 1 and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        env_job_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        env_task_min = os.environ.get("SLURM_ARRAY_TASK_MIN")
        env_task_max = os.environ.get("SLURM_ARRAY_TASK_MAX")
        try:
            args.job_id = int(env_job_id)
            if env_task_min is not None and env_task_max is not None:
                args.total_jobs = int(env_task_max) - int(env_task_min) + 1
            elif env_task_max is not None:
                # Assume min=0 if only max is present
                args.total_jobs = int(env_task_max) + 1
        except ValueError:
            pass

    # --- 1. Prepare all sub-experiments from the EXPERIMENTS list ---
    all_sub_experiments = []
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        for sub_exp in exp["subexperiments"]:
            sub_label = sub_exp["label"]

            current_config = copy.deepcopy(CONFIG)
            current_config.update(sub_exp["overrides"])

            # Create path for CSV logger
            results_folder = current_config.get("results_folder", "Experiments_Folder")
            exp_folder_path = os.path.join(results_folder, exp_name)

            sanitized_label = "".join(
                c for c in sub_label if c.isalnum() or c in (" ", "_")
            ).rstrip()
            csv_filename = f"{sanitized_label.replace(' ', '_')}.csv"
            csv_log_path = os.path.join(exp_folder_path, csv_filename)

            all_sub_experiments.append(
                {
                    "exp_name": exp_name,
                    "sub_label": sub_label,
                    "config": current_config,
                    "csv_log_path": csv_log_path,
                }
            )

    # --- 2. Slice experiments for this specific SLURM job ---
    if args.total_jobs > 1:
        print(
            f"Job Array Mode: Running slice {args.job_id} of {args.total_jobs} total jobs."
        )
        num_exps = len(all_sub_experiments)
        exps_per_job = (num_exps + args.total_jobs - 1) // args.total_jobs
        start_idx = args.job_id * exps_per_job
        end_idx = min(start_idx + exps_per_job, num_exps)
        my_sub_experiments = all_sub_experiments[start_idx:end_idx]
        print(
            f"This node will run {len(my_sub_experiments)} experiments (indices {start_idx} to {end_idx-1})."
        )
    else:
        my_sub_experiments = all_sub_experiments
        print("Single job mode: this node will run all experiments.")

    # --- 3. Run this node's assigned experiments one by one using DDP ---
    world_size = torch.cuda.device_count()
    master_addr = os.environ.get(
        "HOSTNAME", "localhost"
    )  # Get hostname from SLURM environment if available

    for sub_exp_details in my_sub_experiments:
        run_name = sub_exp_details["sub_label"]
        config = sub_exp_details["config"]
        csv_path = sub_exp_details["csv_log_path"]

        # Find a free port for each DDP run to avoid conflicts
        master_port = find_free_port()

        # Launch DDP processes for this single experiment
        mp.spawn(
            run_ddp_worker,
            args=(world_size, master_addr, master_port, config, run_name, csv_path),
            nprocs=world_size,
            join=True,
        )

    print(f"\nNode {args.job_id} has completed all its assigned experiments.")
