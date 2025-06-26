# experiments.py
import wandb
import datetime
import csv
import time
import os
import numpy as np
from core import *  # Import everything from core
from config_generator import chinchilla_scale  # Import the scaling function
import copy
import argparse
import multiprocessing as mp


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
            project_name = f"{project_name_base}-{exp_name}"

            print(f"\nRunning sub-experiment: {exp_name} -> {sub_label}")

            with wandb.init(
                project=project_name, config=config, name=sub_label, reinit=True
            ) as run:
                training_results = train(gpu_id=gpu_id, csv_log_path=csv_log_path)

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
                run.finish()

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

    # Base configuration for all experiments
    base_config = {
        "dataset": "c4_subset",
        "batch_size": 32,  # physical batch size 256
        "learning_rate": 0.001 * math.sqrt(4),
        "min_lr": 1e-5,
        "lr_schedule": "cosine",
        "warmup_epochs": 0,
        "warmup_epochs_frac": 0.1,
        "weight_decay": 0.1,
        "hidden_dim": 64,  # Base hidden dimension
        "num_layers": 4,  # Base number of layers
        "num_heads": 4,
        "dropout": 0.0,
        "seq_length": 128,
        "wikitext_limit": 5 * 10**7,
        "pos_encoding": "rotary",
        "init_scheme": "transformer_scaled",
        "stride": 64,
        "pin_memory": True,
        "compile": False,
        "prefetch_factor": 8,
        "min_epochs": 1,
        "max_epochs": 1,
        "use_gradient_clipping": True,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.0,
        "gradient_accumulation_steps": 16,
        "optimizer": "adamw",
        "activation": "gelu",
        "norm_type": "layer",
        "results_folder": "Former_Experiments_Folder",
        "csv_log_interval": 50,
        "seed": 789,
    }

    # ====================================================================
    # EXPERIMENT DEFINITIONS
    # ====================================================================

    # Activation Function Experiments
    ACTIVATION_EXPERIMENTS = [
        {
            "name": "Activation_Functions_Comparison",
            "subexperiments": [
                {"label": "GELU", "overrides": {"activation": "gelu"}},
                {"label": "ReLU", "overrides": {"activation": "relu"}},
                {"label": "SwiGLU", "overrides": {"activation": "swiglu"}},
            ],
        },
    ]

    # Hidden Dimension Scaling Experiments (simple override approach)
    HIDDEN_DIM_EXPERIMENTS = [
        {
            "name": "Hidden_Dim_Scaling",
            "subexperiments": [
                {
                    "label": "16d",
                    "overrides": {
                        "hidden_dim": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                        "learning_rate": 0.001,
                        "wikitext_limit": 16205120 * 4,
                    },
                },
                {
                    "label": "32d",
                    "overrides": {
                        "hidden_dim": 32,
                        "num_layers": 3,
                        "num_heads": 2,
                        "learning_rate": 0.001,
                        "wikitext_limit": 32901760 * 4,
                    },
                },
                {
                    "label": "64d",
                    "overrides": {
                        "hidden_dim": 64,
                        "num_layers": 4,
                        "num_heads": 4,
                        "learning_rate": 0.002,
                        "wikitext_limit": 68261120 * 4,
                    },
                },
                {
                    "label": "96d",
                    "overrides": {
                        "hidden_dim": 96,
                        "num_layers": 6,
                        "num_heads": 6,
                        "learning_rate": 0.0024,
                        "wikitext_limit": 109764480 * 4,
                    },
                },
            ],
        },
    ]
    HIDDEN_DIM_EXPERIMENTS_NO_ROTARY = [
        {
            "name": "Hidden_Dim_Scaling_No_Rotary",
            "subexperiments": [
                {
                    "label": "16d_no_rotary",
                    "overrides": {
                        "hidden_dim": 16,
                        "num_layers": 2,
                        "num_heads": 1,
                        "learning_rate": 0.001,
                        "wikitext_limit": 16205120 * 4,
                        "pos_encoding": "sinusoidal",
                    },
                },
                {
                    "label": "32d_no_rotary",
                    "overrides": {
                        "hidden_dim": 32,
                        "num_layers": 3,
                        "num_heads": 2,
                        "learning_rate": 0.001,
                        "wikitext_limit": 32901760 * 4,
                        "pos_encoding": "sinusoidal",
                    },
                },
                {
                    "label": "64d_no_rotary",
                    "overrides": {
                        "hidden_dim": 64,
                        "num_layers": 4,
                        "num_heads": 4,
                        "learning_rate": 0.002,
                        "wikitext_limit": 68261120 * 4,
                        "pos_encoding": "sinusoidal",
                    },
                },
                {
                    "label": "96d_no_rotary",
                    "overrides": {
                        "hidden_dim": 96,
                        "num_layers": 6,
                        "num_heads": 6,
                        "learning_rate": 0.0024,
                        "wikitext_limit": 109764480 * 4,
                        "pos_encoding": "sinusoidal",
                    },
                },
            ],
        },
    ]

    # {
    #                 "label": "128d",
    #                 "overrides": {
    #                     "hidden_dim": 128,
    #                     "num_layers": 8,
    #                     "num_heads": 8,
    #                     "learning_rate": 0.0028,
    #                     "wikitext_limit": 160115200 * 4,
    #                 },
    #             },

    # Chinchilla-Scaled Hidden Dimension Experiments (using config generator)
    def create_chinchilla_scaled_experiments():
        """Generate properly scaled experiments using chinchilla_scale function"""

        # Define the hidden dimensions to test
        hidden_dims = [16, 32, 64, 128]

        # Generate scaled configs
        scaled_configs = chinchilla_scale(base_config, hidden_dims)

        # Create subexperiments from the scaled configs
        subexperiments = []
        for i, config in enumerate(scaled_configs):
            hidden_dim = hidden_dims[i]

            # Add any experiment-specific overrides
            config.update(
                {
                    "results_folder": "Former_Experiments_Folder",
                    "csv_log_interval": 50,
                    "seed": 789,
                }
            )

            subexperiments.append(
                {
                    "label": f"{hidden_dim}d",
                    "config": config,  # Use the full generated config
                }
            )

            # Print the generated config for inspection
            print(f"\nGenerated config for {hidden_dim}d:")
            print(
                f"  Architecture: {config['hidden_dim']}d x {config['num_layers']}L x {config['num_heads']}H"
            )
            print(f"  Batch size: {config['batch_size']}")
            print(f"  Learning rate: {config['learning_rate']:.2e}")
            print(f"  Max epochs: {config['max_epochs']}")
            print(f"  Target tokens: {config.get('target_tokens', 'N/A')}")

        return [{"name": "Chinchilla_Experiments", "subexperiments": subexperiments}]

    # Generate the Chinchilla-scaled experiments
    # CHINCHILLA_SCALED_EXPERIMENTS = create_chinchilla_scaled_experiments()

    # ====================================================================
    # SELECT WHICH EXPERIMENTS TO RUN
    # ====================================================================

    # Choose which experiment type to run:
    EXPERIMENTS = (
        HIDDEN_DIM_EXPERIMENTS_NO_ROTARY + HIDDEN_DIM_EXPERIMENTS
    )  # Or use simple hidden dim scaling
    # EXPERIMENTS = ACTIVATION_EXPERIMENTS       # Or use activation experiments
    # EXPERIMENTS = CHINCHILLA_SCALED_EXPERIMENTS  # Use Chinchilla scaling

    # ====================================================================
    # EXPERIMENT PROCESSING
    # ====================================================================

    # Prepare all sub-experiments
    all_sub_experiments = []

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

            all_sub_experiments.append(
                {
                    "exp_name": exp_name,
                    "sub_label": sub_label,
                    "config": current_config,
                    "csv_log_path": csv_log_path,
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
