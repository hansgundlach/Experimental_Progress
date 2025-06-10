import torch
import torch.distributed as dist
import os
import time
import copy
import math
from lstm_training import train_model

# Configuration
# has 1.66 total params
CONFIG = {
    "data_path": "../Datasets/wikitext.txt",
    "tokenizer_path": "../gpt2_tokenizer",
    "max_characters": 5 * 1e4,  # Maximum number of characters to use from dataset
    "sequence_length": 128,
    "batch_size": 256,  # Keep physical batch size small
    "hidden_size": 16,
    "num_layers": 2,
    "dropout": 0.0,  # dropout zer here to match transformer but may need to adjust for LSTM
    "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
    "lr_schedule": "cosine",
    "step_size": 10,
    "gamma": 0.1,
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
    # NEW: Data loading optimization settings
    "num_workers": "auto",  # Will be set automatically based on CPU cores
    "pin_memory": True,  # Faster GPU memory transfer
    "persistent_workers": True,  # Keep data loading workers alive between epochs
    "prefetch_factor": 4,  # Number of batches to prefetch per worker
    # NEW: Mixed precision settings
    "use_amp": False,  # Enable Automatic Mixed Precision
    "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
    # NEW: Gradient accumulation settings
    "gradient_accumulation_steps": 2,  # For tracking only
    # NEW: whether to compile the model (PyTorch 2.0+)
    "use_compile": False,
    "seed": 789,
    "optimizer": "adamw",  # NEW: choose from "adam", "adamw", or "sgd"
    "weight_decay": 0.1,
    "stride": 64,  # NEW: sliding-window stride to match transformer
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
EXPERIMENTS = [
    {
        "name": "Diag_experiment_1",
        "subexperiments": [
            {
                "label": "My label for sub experiment 1",
                "overrides": {"learning_rate": 1e-2},
            },
            {
                "label": "My label for sub experiment 2",
                "overrides": {"sequence_length": 150},
            },
            {
                "label": "My label for sub experiment 3",
                "overrides": {"lr_schedule": "cosine", "step_size": 20},
            },
        ],
    },
    # – Add more experiments here, e.g.
    # {
    #   "name": "Another_experiment",
    #   "subexperiments": [
    #     { "label": "foo", "overrides": {...} },
    #     …
    #   ]
    # },
]
# ============================================================================


def run_experiment_suite(base_config, local_rank=0):
    """Runs a suite of experiments with different configurations."""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    overall_results = {}

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        sub_experiments = exp["subexperiments"]
        if is_main_process:
            print(f"\n{'='*20} Running Experiment: {exp_name} {'='*20}")
        overall_results[exp_name] = {}

        for sub in sub_experiments:
            sub_exp_name = sub["label"]
            overrides = sub["overrides"]
            if is_main_process:
                print(f"\n--- Running Sub-Experiment: {sub_exp_name} ---")

            current_config = copy.deepcopy(base_config)
            current_config.update(overrides)

            # Group them under a sub‐project if you like:
            current_config["wandb_project"] = (
                f"{base_config['wandb_project']}-{exp_name}"
            )

            if is_main_process:
                print("Configuration overrides:")
                for key, value in overrides.items():
                    print(f"  {key}: {value}")

            _, results = train_model(current_config, local_rank, run_name=sub_exp_name)
            overall_results[exp_name][sub_exp_name] = results

            if is_main_process:
                time.sleep(2)  # cleaner logging

    # Print summary of results
    if is_main_process:
        print(f"\n{'='*20} Experiment Suite Summary {'='*20}")
        for exp_name, sub_results in overall_results.items():
            print(f"\nResults of '{exp_name}':")
            for sub_exp_name, result_metrics in sub_results.items():
                print(f"  '{sub_exp_name}':")
                print(
                    f"    - Final Training Loss: {result_metrics['final_train_loss']:.4f}"
                )
                print(
                    f"    - Final Validation Loss: {result_metrics['final_val_loss']:.4f}"
                )
                print(
                    f"    - Total FLOPs (Profiler): {result_metrics['total_flops_profiler']:.2e}"
                )
                print(
                    f"    - Total FLOPs (Theoretical): {result_metrics['total_flops_theoretical']:.2e}"
                )


if __name__ == "__main__":
    # Check if we're running in distributed mode
    # Look for SLURM variables
    use_ddp = (
        "SLURM_JOB_ID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1
    ) or ("LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1)
    local_rank = 0

    if use_ddp:
        # Get local rank from SLURM or PyTorch
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        torch.cuda.set_device(local_rank)
        # Make sure these environment variables are set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(
                os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1))
            ),
            rank=int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))),
        )
        CONFIG["device"] = f"cuda:{local_rank}"
        print(f"Starting LSTM training under DDP: rank {local_rank}")
    else:
        # Single GPU mode
        CONFIG["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Starting LSTM training in single GPU mode…")

    print(f"Configuration: {CONFIG}")

    run_experiment_suite(CONFIG, local_rank)
