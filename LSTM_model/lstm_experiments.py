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

CONFIG = get_lstm_base_config()


# ====================================================================
# EXPERIMENT SELECTION
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
    # ONLY generate from the last job to avoid duplicate generation
    is_last_job = (args.job_id == args.total_jobs - 1) or (args.total_jobs == 1)

    if is_last_job:  # Only last job generates summary
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
                print(f"(Job {args.job_id} is the last job - generating summary)")
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
