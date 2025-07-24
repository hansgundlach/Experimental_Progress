import torch
import torch.distributed as dist
import os
import time
import copy
import math
from lstm_training import train_model
import argparse
import torch.multiprocessing as mp
from socket import socket

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
    "seed": 789,
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

# – Add more experiments here, e.g.
# {
#   "name": "Another_experiment",
#   "subexperiments": [
#     { "label": "foo", "overrides": {...} },
#     …
#   ]
# },
# ]
# =====
LSTM_HIDDEN_DIM_EXPERIMENTS = [
    {
        "name": "LSTM_Hidden_Dim_Scaling",
        "subexperiments": [
            {
                "label": "LSTM_16d_123",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "hidden_size": 16,
                    "max_characters": 129e6,
                    "seed": 123,
                },
            },
            {
                "label": "LSTM_24d_123",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "LSTM_32d_123",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 258.6e6,
                    "seed": 123,
                    "hidden_size": 32,
                },
            },
            {
                "label": "LSTM_48d_123",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 388.8e6,
                    "seed": 123,
                    "hidden_size": 48,
                },
            },
        ],
    },
]

LSTM_HYPER_PARAM_EXPERIMENTS = [
    {
        "name": "LSTM_Hyper_Param_Experiments",
        "subexperiments": [
            {
                "label": "LSTM_24d_123_no_layer_norm",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "use_layer_norm": False,
                },
            },
            {
                "label": "LSTM_24d_123_standard",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                },
            },
            {
                "label": "LSTM_24d_123_no_long_stride",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 193.7e6,
                    "seed": 123,
                    "hidden_size": 24,
                    "stride": 128,
                },
            },
            {
                "label": "LSTM_24d_123_no_half_tokens",
                "overrides": {
                    "learning_rate": 3 * 1e-3,
                    "max_characters": 193.7e6 / 2,
                    "seed": 123,
                    "hidden_size": 24,
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
# ============================================================================
EXPERIMENTS = LSTM_HYPER_PARAM_EXPERIMENTS


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
