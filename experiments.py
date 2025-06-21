# experiments.py
import wandb
import datetime
import csv
import time
import os
import numpy as np
from core import *  # Import everything from core
import copy


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

    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name_base = f"transformer_experiments_{timestamp}"

    # Detect available compute resources
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("Warning: No GPUs detected. Running on CPU.")

    # Base configuration for all experiments
    base_config = {
        "dataset": "wikitext",
        "batch_size": 256,
        "learning_rate": 0.001 * math.sqrt(4),
        "min_lr": 1e-5,
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
        "warmup_epochs_frac": 0.1,
        "weight_decay": 0.1,
        "hidden_dim": 16,  # reduced from 64 → yields ~1.6M params
        "num_layers": 2,  # shallow network
        "num_heads": 4,  # must divide hidden_di
        "dropout": 0.0,
        "seq_length": 128,
        "wikitext_limit": 5 * 10**5,
        "pos_encoding": "sinusoidal",
        "init_scheme": "xavier_uniform",
        "stride": 64,
        "pin_memory": True,
        "compile": False,
        "prefetch_factor": 8,
        "min_epochs": 5,
        "max_epochs": 5,
        "use_gradient_clipping": True,
        "gradient_clip_val": 1.0,
        "label_smoothing": 0.0,
        "gradient_accumulation_steps": 4,
        "optimizer": "adamw",
        "activation": "gelu",
        "norm_type": "layer",
        # NEW: CSV logging settings
        "results_folder": "Former_Experiments_Folder",
        "csv_log_interval": 100,  # Log every N optimizer steps
        "seed": 789,
    }

    # Define the experiment suite
    EXPERIMENTS = [
        {
            "name": "Diag_experiment_1",
            "subexperiments": [
                {
                    "label": "High_Learning_Rate",
                    "overrides": {"learning_rate": 1e-2},
                },
                {
                    "label": "Longer_Sequence_Length",
                    "overrides": {"seq_length": 256},
                },
                {
                    "label": "Inverse_Sqrt_LR_Schedule",
                    "overrides": {"lr_schedule": "inverse_sqrt"},
                },
                {"label": "Baseline_Run", "overrides": {}},
            ],
        },
        {
            "name": "Activation_Functions_Comparison",
            "subexperiments": [
                {"label": "GELU", "overrides": {"activation": "gelu"}},
                {"label": "ReLU", "overrides": {"activation": "relu"}},
                {"label": "SwiGLU", "overrides": {"activation": "swiglu"}},
            ],
        },
    ]

    # Prepare all sub-experiments
    all_sub_experiments = []
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        for sub_exp in exp["subexperiments"]:
            sub_label = sub_exp["label"]

            # Create config for this specific run
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

    # Run experiments
    overall_results = {}
    if n_gpus > 1:
        total_start_time = time.time()
        print(
            f"\nRunning {len(all_sub_experiments)} total sub-experiments across {n_gpus} GPUs"
        )
        processes = []
        results_queue = mp.Queue()
        experiments_per_gpu = (len(all_sub_experiments) + n_gpus - 1) // n_gpus

        for gpu_id in range(n_gpus):
            start_idx = gpu_id * experiments_per_gpu
            end_idx = min(start_idx + experiments_per_gpu, len(all_sub_experiments))
            gpu_experiments = all_sub_experiments[start_idx:end_idx]

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
            gpu_id, all_sub_experiments, project_name_base
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

    print("\nTOTAL SUB-EXPERIMENTS CREATED:")
    print(f"Number of sub-experiments: {len(all_sub_experiments)}")
