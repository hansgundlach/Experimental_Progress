import torch
import os
import time
import copy
import math
import sys

# Ensure local directory is on sys.path for relative imports when launched by Slurm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lstm_training import train_model
from lstm_experiment_definitions import (
    GRAND_EXPERIMENT,
)
import argparse
from typing import List, Union, Sequence
from lstm_experiment_utils import (
    get_lstm_base_config,
    create_multi_lr_lstm_experiments,
    create_multi_seed_lstm_experiments,
    create_multi_lr_experiments,
    create_multi_seed_experiments,
)
# Import LR sweep summary function from parent experiment_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import generate_lr_sweep_summary

# Configuration
# has 1.66 total params
# CONFIG = {
#     "data_path": "../Datasets/c4_subset.txt",
#     "tokenizer_path": "../gpt2_tokenizer",
#     "max_characters": 5 * 1e7,  # Maximum number of characters to use from dataset
#     "sequence_length": 128,
#     "batch_size": 32,  # Keep physical batch size small, has no effect on model
#     "hidden_size": 16,
#     "num_layers": 2,
#     "dropout": 0.0,  # dropout zer here to match transformer but may need to adjust for LSTM
#     "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
#     "lr_schedule": "cosine",
#     "step_size": 10,
#     "gamma": 0.1,  # parameter usedf for stepLR step decay
#     "num_epochs": 1,
#     "train_split": 0.8,
#     "val_split": 0.1,
#     "test_split": 0.1,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "wandb_project": "lstm-language-modeling",
#     "wandb_offline": True,
#     "print_every": 100,  # Print loss every N batches
#     # Gradient clipping settings
#     "use_gradient_clipping": True,
#     "gradient_clip_val": 1.0,
#     # NEW: CSV logging settings
#     "results_folder": "Experiments_Folder",
#     "csv_log_interval": 20,
#     # NEW: Data loading optimization settings
#     "num_workers": "auto",  # Will be set automatically based on CPU cores
#     "pin_memory": True,  # Faster GPU memory transfer
#     "persistent_workers": True,  # Keep data loading workers alive between epochs
#     "prefetch_factor": 4,  # Number of batches to prefetch per worker
#     # NEW: Mixed precision settings
#     "use_amp": False,  # Enable Automatic Mixed Precision
#     "amp_opt_level": "O1",  # Not used with native AMP, but kept for reference
#     # NEW: Gradient accumulation settings
#     "gradient_accumulation_steps": 16,  # For tracking only
#     # NEW: whether to compile the model (PyTorch 2.0+)
#     "use_compile": False,
#     "seed": 123,
#     "optimizer": "adamw",  # NEW: choose from "adam", "adamw", or "sgd"
#     "weight_decay": 0.01,
#     "stride": 128,  # NEW: sliding-window stride to match transformer
#     # Add three separate variational dropout parameters
#     "input_dropout": 0.2,  # Applied to embeddings
#     "hidden_dropout": 0.1,  # Applied between LSTM layers
#     "output_dropout": 0.2,  # Applied before final linear layer
#     "use_layer_norm": True,  # Enable/disable LayerNorm
#     "layer_norm_position": "output",  # Options: "input", "output", "both", "gates"
#     "use_mup": True,
#     "mup_base_width": 16,
#     "tie_embeddings": True,  # Enable weight tying by default
# }
CONFIG = get_lstm_base_config()


# ====================================================================
# EXPERIMENT SELECTION
# ====================================================================

# The EXPERIMENTS variable is imported from lstm_experiment_definitions.py
# You can override it here if needed, or modify the default in the definitions file

# Example overrides:
# EXPERIMENTS = MUP_SCALING_EXPERIMENTS
# EXPERIMENTS = LSTM_LR_EXPERIMENTS
# EXPERIMENTS = subset_experiments(LSTM_SGD_MUP_SCALING, {"lstm_16d_sgd_mup", "lstm_24d_sgd_mup"})


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================


EXPERIMENTS = GRAND_EXPERIMENT


if __name__ == "__main__":
    print("Starting LSTM experiments...")
    parser = argparse.ArgumentParser(description="Run LSTM Experiments")
    parser.add_argument("--job_id", type=int, default=0, help="SLURM job array ID")
    parser.add_argument(
        "--total_jobs", type=int, default=1, help="Total SLURM jobs in array"
    )
    args = parser.parse_args()
    print(f"Arguments parsed: job_id={args.job_id}, total_jobs={args.total_jobs}")

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
    print(f"Building experiments from {len(EXPERIMENTS)} experiment groups...")
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

            # Store the folder name (exp_name) for wandb project naming
            folder_name = exp_name

            all_sub_experiments.append(
                {
                    "exp_name": exp_name,
                    "sub_label": sub_label,
                    "config": current_config,
                    "csv_log_path": csv_log_path,
                    "folder_name": folder_name,
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

    # --- 3. Run this node's assigned experiments one by one using single-GPU training ---
    for sub_exp_details in my_sub_experiments:
        run_name = sub_exp_details["sub_label"]
        config = sub_exp_details["config"]
        csv_path = sub_exp_details["csv_log_path"]

        print(
            f"\nStarting LSTM training for: {run_name} at {time.strftime('%H:%M:%S')}"
        )
        training_start_time = time.time()

        # Run single-GPU training directly (no DDP, no multiprocessing)
        train_model(
            config=config,
            run_name=run_name,
            csv_log_path=csv_path,
            folder_name=sub_exp_details["folder_name"],
        )

        training_elapsed = time.time() - training_start_time
        print(f"LSTM training completed for: {run_name}")
        print(f"Training time: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
        print(f"Completed LSTM experiment: {run_name}")

    print(f"\nNode {args.job_id} has completed all its assigned experiments.")
    
    # Generate LR sweep summary if this was a learning rate sweep experiment
    # Generate summary regardless of job count - each job will generate the same summary safely
    if True:  # Always attempt to generate summary if experiments have generate_summary=True
        try:
            # Check if any experiment has summary info (indicates lr sweep with generate_summary=True)
            summary_info = None
            for exp in GRAND_EXPERIMENT:
                if hasattr(exp, '_summary_info') or '_summary_info' in exp:
                    summary_info = exp.get('_summary_info')
                    break
            
            if summary_info and summary_info.get('generate_summary', False):
                print("\n" + "="*50)
                print("Generating Learning Rate Sweep Summary...")
                print("="*50)
                
                # Get the base results folder from LSTM config
                lstm_base_config = get_lstm_base_config()
                results_base_folder = lstm_base_config.get("results_folder", "new_experiments_folder_1")
                
                summary_path = generate_lr_sweep_summary(summary_info, results_base_folder)
                if summary_path:
                    print(f"✅ LR sweep summary generated successfully!")
                    print(f"📄 Summary saved to: {summary_path}")
                else:
                    print("⚠️  LR sweep summary generation failed or no valid results found")
            else:
                # Check if experiment name suggests it's an LR sweep (fallback detection)
                for exp in GRAND_EXPERIMENT:
                    if "_lr_sweep" in exp["name"]:
                        print(f"Note: Detected LR sweep experiment '{exp['name']}' but generate_summary was not enabled.")
                        print("To enable automatic summary generation, use generate_summary=True in create_multi_lr_experiments()")
                        break
        except Exception as e:
            print(f"Error during LR sweep summary generation: {e}")
            import traceback
            traceback.print_exc()
