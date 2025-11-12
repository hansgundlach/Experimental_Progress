# experiments.py
import wandb
import datetime
import csv
import time
import os
import numpy as np
from core import *  # Import everything from core
import copy
import argparse
import multiprocessing as mp
from experiment_utils import (
    create_multi_seed_experiments,
    create_multi_lr_experiments,
    calculate_transformer_params,
    gen_experim,
    get_base_config,
    generate_lr_sweep_summary,
)
from experiment_definitions import (
    GRAND_EXPERIMENT,
)
from non_scaling_experim_def import (
    GRAND_VARIATION_EXPERIMENTS,
)

EXPERIMENTS = GRAND_EXPERIMENT


def run_experiments_on_gpu(gpu_id, sub_experiments, project_name_base):
    print(f"\nGPU {gpu_id} STARTING:")
    print(f"Number of experiments on GPU: {len(sub_experiments)}")
    start_time = time.time()
    results = {}

    for sub_exp in sub_experiments:
        try:
            config = sub_exp["config"]
            exp_name = sub_exp["exp_name"]
            sub_label = sub_exp["sub_label"]
            csv_log_path = sub_exp["csv_log_path"]
            folder_name = sub_exp["folder_name"]
            # Use folder name as the wandb project name instead of timestamp-based naming
            project_name = folder_name

            print(f"\nRunning sub-experiment: {exp_name} -> {sub_label}")

            print(f"Starting training for: {sub_label} at {time.strftime('%H:%M:%S')}")
            training_start_time = time.time()

            with wandb.init(
                project=project_name, config=config, name=sub_label, reinit=True
            ) as run:
                training_results = train(gpu_id=gpu_id, csv_log_path=csv_log_path)

                training_elapsed = time.time() - training_start_time
                print(f"Training completed for: {sub_label}")
                print(
                    f"Training time: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)"
                )
                print(
                    f"Final validation loss: {training_results['final_val_loss']:.4f}"
                )

                if exp_name not in results:
                    results[exp_name] = {}
                results[exp_name][sub_label] = training_results

                wandb.log(
                    {
                        "final_train_loss": training_results["final_train_loss"],
                        "final_val_loss": training_results["final_val_loss"],
                        "best_val_loss": training_results["best_val_loss"],
                    }
                )

            print(f"Completed sub-experiment: {exp_name} -> {sub_label}")
            print(f"Results: {training_results}")

        except Exception as e:
            print(f"Error in sub-experiment {sub_exp} on GPU {gpu_id}: {str(e)}")
            exp_name = sub_exp.get("exp_name", "unknown_exp")
            sub_label = sub_exp.get("sub_label", "unknown_sub_exp")
            if exp_name not in results:
                results[exp_name] = {}
            results[exp_name][sub_label] = {
                "final_train_loss": float("nan"),
                "final_val_loss": float("nan"),
                "best_val_loss": float("nan"),
                "total_flops_profiler": 0,
                "total_flops_theoretical": 0,
            }

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id} completed all experiments in {elapsed:.2f} seconds")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Transformer Experiments")
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="The ID of the current job in a SLURM job array.",
    )
    parser.add_argument(
        "--total_jobs",
        type=int,
        default=1,
        help="The total number of jobs in the SLURM job array.",
    )
    args = parser.parse_args()

    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name_base = f"transformer_experiments_{timestamp}"

    # Detect available compute resources
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("Warning: No GPUs detected. Running on CPU.")

    # Get base configuration from utilities
    base_config = get_base_config()

    # Generate the Chinchilla-scaled experiments
    # CHINCHILLA_SCALED_EXPERIMENTS = create_chinchilla_scaled_experiments()

    # ====================================================================
    # SELECT WHICH EXPERIMENTS TO RUN
    # ====================================================================

    # Choose which experiment type to run:
    # EXPERIMENTS = (
    #     HIDDEN_DIM_EXPERIMENTS_NO_ROTARY_123 + HIDDEN_DIM_EXPERIMENTS_123
    # )  # Or use simple hidden dim scaling
    # EXPERIMENTS = ACTIVATION_EXPERIMENTS       # Or use activation experiments
    # EXPERIMENTS = CHINCHILLA_SCALED_EXPERIMENTS  # Use Chinchilla scaling
    def subset_experiments(experiment_list, wanted_labels):
        """Return only the sub-experiments whose 'label' is in wanted_labels,
        keeping each group's original name unchanged."""
        result = []
        for exp in experiment_list:
            picked = [
                se for se in exp["subexperiments"] if se["label"] in wanted_labels
            ]
            if picked:
                # keep exp['name'] exactly as-is
                result.append(
                    {
                        "name": exp["name"],
                        "subexperiments": picked,
                    }
                )
        return result

    # choose relevant experiments

    # EXPERIMENTS = GRAND_VARIATION_EXPERIMENTS

    # Initialize the list to store all sub-experiments
    all_sub_experiments = []

    # EXPERIMENTS =
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        for sub_exp in exp["subexperiments"]:
            sub_label = sub_exp["label"]

            # Check if this subexperiment uses a pre-generated config or overrides
            if "config" in sub_exp:
                # Use the pre-generated config (for Chinchilla-scaled experiments)
                current_config = sub_exp["config"]
            else:
                # Use base config with overrides (for simple experiments)
                current_config = copy.deepcopy(base_config)
                current_config.update(sub_exp["overrides"])

            # Create path for CSV logger
            results_folder = current_config.get(
                "results_folder", "Former_Experiments_Folder"
            )
            exp_folder_path = os.path.join(results_folder, exp_name)
            # Sanitize the label to create a valid filename
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

    # DEBUG: Print what we're planning to do
    print(f"\n{'='*50}")
    print(f"DEBUG: Total sub-experiments created: {len(all_sub_experiments)}")
    print(f"DEBUG: Experiment paths that will be created:")
    for i, sub_exp in enumerate(all_sub_experiments):
        print(f"  {i}: {sub_exp['csv_log_path']}")
        print(f"      Exp: {sub_exp['exp_name']} -> {sub_exp['sub_label']}")
    print(f"{'='*50}\n")

    # Slice the experiments based on the job array ID
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
            f"This job will run {len(my_sub_experiments)} experiments (indices {start_idx} to {end_idx-1})."
        )
        print(
            f"DEBUG: This job will process experiments: {[exp['sub_label'] for exp in my_sub_experiments]}"
        )
    else:
        my_sub_experiments = all_sub_experiments
        print(f"DEBUG: Single job mode - processing all experiments")

    # Run experiments
    overall_results = {}
    if n_gpus > 1:
        total_start_time = time.time()
        print(
            f"\nRunning {len(my_sub_experiments)} total sub-experiments across {n_gpus} GPUs on this node"
        )
        processes = []
        results_queue = mp.Queue()
        experiments_per_gpu = (len(my_sub_experiments) + n_gpus - 1) // n_gpus

        for gpu_id in range(n_gpus):
            start_idx = gpu_id * experiments_per_gpu
            end_idx = min(start_idx + experiments_per_gpu, len(my_sub_experiments))
            gpu_experiments = my_sub_experiments[start_idx:end_idx]

            if not gpu_experiments:
                continue

            print(
                f"\nGPU {gpu_id} assigned sub-experiments {start_idx} to {end_idx-1}:"
            )
            print(f"Labels: {[exp['sub_label'] for exp in gpu_experiments]}")

            p = mp.Process(
                target=lambda q, gid, exps, proj: q.put(
                    (gid, run_experiments_on_gpu(gid, exps, proj))
                ),
                args=(results_queue, gpu_id, gpu_experiments, project_name_base),
            )
            p.daemon = False
            processes.append(p)
            p.start()

        # Collect results from all processes
        print("\nCollecting results from GPUs:")
        for _ in range(len(processes)):
            gpu_id, gpu_results = results_queue.get()
            print(f"\nReceived results from GPU {gpu_id}:")
            for exp_name, sub_results in gpu_results.items():
                if exp_name not in overall_results:
                    overall_results[exp_name] = {}
                overall_results[exp_name].update(sub_results)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        total_elapsed = time.time() - total_start_time
        print(f"\nAll experiments completed in {total_elapsed:.2f} seconds")

    else:
        # Single GPU or CPU setup
        gpu_id = 0 if torch.cuda.is_available() else None
        print(f"Running on single device (GPU: {gpu_id})")
        overall_results = run_experiments_on_gpu(
            gpu_id, my_sub_experiments, project_name_base
        )

    # Print final summary of results
    print(f"\n{'='*20} Experiment Suite Summary {'='*20}")
    for exp_name, sub_results in overall_results.items():
        print(f"\nResults of '{exp_name}':")
        for sub_label, result_metrics in sub_results.items():
            print(f"  '{sub_label}':")
            if result_metrics and "final_val_loss" in result_metrics:
                print(
                    f"    - Final Training Loss: {result_metrics['final_train_loss']:.4f}"
                )
                print(
                    f"    - Final Validation Loss: {result_metrics['final_val_loss']:.4f}"
                )
                print(
                    f"    - Best Validation Loss: {result_metrics['best_val_loss']:.4f}"
                )
                print(
                    f"    - Total FLOPs (Profiler): {result_metrics['total_flops_profiler']:.2e}"
                )
                print(
                    f"    - Total FLOPs (Theoretical): {result_metrics['total_flops_theoretical']:.2e}"
                )
            else:
                print("    - Experiment failed to produce results.")

    print("\nTOTAL SUB-EXPERIMENTS CREATED (ACROSS ALL JOBS):")
    print(f"Number of sub-experiments: {len(all_sub_experiments)}")
    if args.total_jobs > 1:
        print(f"THIS JOB RAN: {len(my_sub_experiments)} sub-experiments.")

    # Generate LR sweep summary if this was a learning rate sweep experiment
    # Generate summary regardless of job count - each job will generate the same summary safely
    if (
        True
    ):  # Always attempt to generate summary if experiments have generate_summary=True
        try:
            # Check if any experiment has summary info (indicates lr sweep with generate_summary=True)
            summary_info = None
            for exp in EXPERIMENTS:
                if hasattr(exp, "_summary_info") or "_summary_info" in exp:
                    summary_info = exp.get("_summary_info")
                    break

            if summary_info and summary_info.get("generate_summary", False):
                print("\n" + "=" * 50)
                print("Generating Learning Rate Sweep Summary...")
                print("=" * 50)

                # Get the base results folder from config
                results_base_folder = base_config.get(
                    "results_folder", "new_experiments_folder_1"
                )

                summary_path = generate_lr_sweep_summary(
                    summary_info, results_base_folder
                )
                if summary_path:
                    print(f"✅ LR sweep summary generated successfully!")
                else:
                    print(
                        "⚠️  LR sweep summary generation failed or no valid results found"
                    )
            else:
                # Check if experiment name suggests it's an LR sweep (fallback detection)
                for exp in EXPERIMENTS:
                    if "_lr_sweep" in exp["name"]:
                        print(
                            f"Note: Detected LR sweep experiment '{exp['name']}' but generate_summary was not enabled."
                        )
                        print(
                            "To enable automatic summary generation, use generate_summary=True in create_multi_lr_experiments()"
                        )
                        break
        except Exception as e:
            print(f"Error during LR sweep summary generation: {e}")
            import traceback

            traceback.print_exc()
