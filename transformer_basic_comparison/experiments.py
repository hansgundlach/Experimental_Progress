# experiments.py
import wandb
import datetime
import csv
import time
import os
import numpy as np
from core import *  # Import everything from core


def run_experiments_on_gpu(gpu_id, experiments, parameter, project_name, base_config):
    print(f"\nGPU {gpu_id} STARTING:")
    print(f"Number of experiments on GPU: {len(experiments)}")
    print("Experiments:", [(exp[parameter], exp["seed"]) for exp in experiments])
    start_time = time.time()
    results = {}

    for exp in experiments:
        try:
            # Create path for CSV logger
            csv_log_path = None
            results_folder = base_config.get("results_folder", "results")
            csv_log_interval = base_config.get("csv_log_interval")

            if results_folder and csv_log_interval:
                exp_folder_name = parameter  # e.g., "activation"
                # Sanitize option value for filename
                option_val_str = str(exp[parameter]).replace("/", "_")
                sub_exp_filename = f"{option_val_str}_seed_{exp['seed']}.csv"
                csv_log_path = os.path.join(
                    results_folder, exp_folder_name, sub_exp_filename
                )

            with wandb.init(
                project=project_name, config={**base_config, **exp}, reinit=True
            ) as run:
                training_results = train(gpu_id=gpu_id, csv_log_path=csv_log_path)

                # Use the current parameter type instead of hardcoding "activation"
                param_value = exp[parameter]
                seed = exp["seed"]
                if param_value not in results:
                    results[param_value] = {}
                results[param_value][seed] = {
                    "final_loss": training_results["val_loss"],
                    "best_loss": training_results["best_val_loss"],
                }

                wandb.log(
                    {
                        "final_val_loss": training_results["val_loss"],
                        "best_val_loss": training_results["best_val_loss"],
                    }
                )
                run.finish()

                print(f"Completed experiment: {param_value} seed {seed}")
                print(
                    f"Results: final_loss={training_results['val_loss']:.4f}, best_loss={training_results['best_val_loss']:.4f}"
                )

        except Exception as e:
            print(f"Error in experiment {exp} on GPU {gpu_id}: {str(e)}")
            param_value = exp[parameter]
            seed = exp["seed"]
            if param_value not in results:
                results[param_value] = {}
            results[param_value][seed] = {
                "final_loss": float("nan"),
                "best_loss": float("nan"),
            }

    elapsed = time.time() - start_time
    print(f"GPU {gpu_id} completed all experiments in {elapsed:.2f} seconds")
    return results


if __name__ == "__main__":

    wandb.login()
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    project_name = f"transformer_experiments_{timestamp}"

    # Detect available compute resources
    n_gpus = torch.cuda.device_count()
    # Base configuration for all experiments

    # best base config
    # base_config = {
    #     "dataset": "wikitext",
    #     "batch_size": 32,  # Larger batches (Chinchilla used big batches) size of each mini batch
    #     "learning_rate": 6e-4,  # Scale with batch size (sqrt scaling)
    #     "min_lr": 1e-5,
    #     "lr_schedule": "cosine_warmup",
    #     "warmup_epochs": 1,
    #     "warmup_epochs_frac": 0.1,  # Shorter warmup
    #     "weight_decay": 0.1,  # Standard Chinchilla weight decay
    #     "hidden_dim": 64,  # Much smaller model
    #     "num_layers": 6,  # Fewer layers
    #     "num_heads": 8,  # Keep heads (64/8 = 8 dim per head)
    #     "dropout": 0.0,  # Chinchilla used little/no dropout
    #     "seq_length": 128,  # Longer sequences (better data efficiency)
    #     "wikitext_limit": 3 * 10**8,
    #     "pos_encoding": "rotary",
    #     "init_scheme": "transformer_scaled",
    #     "stride": 64,  # 50% overlap
    #     "pin_memory": True,
    #     "compile": False,
    #     "prefetch_factor": 8,
    #     "min_epochs": 5,  # MANY more epochs (see data 10-20x)
    #     "max_epochs": 5,
    #     "use_gradient_clipping": True,
    #     "gradient_clip_val": 1.0,
    #     "label_smoothing": 0.0,  # Chinchilla didn't use this
    #     "gradient_accumulation_steps": 4,
    #     "optimizer": "adam",  # Chinchilla used AdamW, I keep getting worse performance with it
    #     "activation": "relu",  # chinchilla used gelu but I get better performance with relu
    #     "norm_type": "layer",
    #     # NEW: CSV logging settings
    #     "results_folder": "Former_Experiments_Folder",
    #     "csv_log_interval": 100,  # Log every N steps
    # }

    # small config used to compare to lstm
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
        "num_heads": 4,  # must divide hidden_dim
        "dropout": 0.0,
        "seq_length": 128,
        "wikitext_limit": 5 * 10**7,
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
    }

    # Setup experiments
    # long_seeds = [42, 123, 789, 1000]
    seeds = [789]

    # comparing activation functions
    # comparison_activation = {
    #     "parameter": "activation",
    #     "options": ["gelu", "relu", "swiglu"],
    #     "base_changes": {
    #         "gelu": {"activation": "gelu"},
    #         "relu": {"activation": "relu"},
    #         "swiglu": {"activation": "swiglu"},
    #         "glu": {"activation": "glu"},
    #     },
    # }

    short_comparison_activation = {
        "parameter": "activation",
        "options": ["glu", "gelu", "relu", "swiglu"],
        "base_changes": {
            "glu": {"activation": "glu"},
            "gelu": {"activation": "gelu"},
            "relu": {"activation": "relu"},
            "swiglu": {"activation": "swiglu"},
        },
    }
    # comparing lr_schedulers
    # comparison_lr_schedule = {
    #     "parameter": "lr_schedule",
    #     "options": [
    #         "cosine",
    #         "cosine_warmup",
    #         "inverse_sqrt",
    #         "one_cycle",
    #         "transformer",
    #     ],
    #     "base_changes": {
    #         "cosine": {"lr_schedule": "cosine"},
    #         "cosine_warmup": {"lr_schedule": "cosine_warmup"},
    #         "inverse_sqrt": {"lr_schedule": "inverse_sqrt"},
    #         "one_cycle": {"lr_schedule": "one_cycle"},
    #         "transformer": {"lr_schedule": "transformer"},
    #     },
    # }
    short_comparison_lr_schedule = {
        "parameter": "lr_schedule",
        "options": [
            "cosine",
            "cosine_warmup",
            "inverse_sqrt",
            "one_cycle",
        ],
        "base_changes": {
            "cosine": {"lr_schedule": "cosine"},
            "cosine_warmup": {"lr_schedule": "cosine_warmup"},
            "inverse_sqrt": {"lr_schedule": "inverse_sqrt"},
            "one_cycle": {"lr_schedule": "one_cycle"},
        },
    }
    comparison_optimizer = {
        "parameter": "optimizer",
        "options": ["adamw", "adam"],
        "base_changes": {
            "adamw": {"optimizer": "adamw"},
            "adam": {"optimizer": "adam"},
        },
    }
    comparison_init_scheme = {
        "parameter": "init_scheme",
        "options": ["transformer_scaled", "xavier_uniform"],
        "base_changes": {
            "transformer_scaled": {"init_scheme": "transformer_scaled"},
            "xavier_uniform": {"init_scheme": "xavier_uniform"},
        },
    }
    comparison_gradient_clipping = {
        "parameter": "use_gradient_clipping",
        "options": [True, False],
        "base_changes": {
            True: {"use_gradient_clipping": True},
            False: {"use_gradient_clipping": False},
        },
    }
    comparison_dropout = {
        "parameter": "dropout",
        "options": [0.0, 0.2],
        "base_changes": {
            0.0: {"dropout": 0.0},
            0.2: {"dropout": 0.2},
        },
    }
    comparison_pos_encoding = {
        "parameter": "pos_encoding",
        "options": ["sinusoidal", "learned", "rotary"],
        "base_changes": {
            "sinusoidal": {"pos_encoding": "sinusoidal"},
            "learned": {"pos_encoding": "learned"},
            "rotary": {"pos_encoding": "rotary"},
        },
    }
    comparison_norms = {
        "parameter": "norm_type",
        "options": ["layer", "rms"],
        "base_changes": {
            "layer": {"norm_type": "layer"},
            "rms": {"norm_type": "rms"},
        },
    }

    comparison_depth = {
        "parameter": "num_layers",
        "options": [2, 4, 8, 10],
        "base_changes": {
            2: {"num_layers": 2},
            4: {"num_layers": 4},
            6: {"num_layers": 6},
            8: {"num_layers": 8},
            10: {"num_layers": 10},
        },
    }

    # Add learning rate sweep for SGD optimizer
    comparison_lr = {
        "parameter": "learning_rate",
        "options": [1e-5, 1e-4, 1e-3, 1e-2],
        "base_changes": {
            1e-5: {"learning_rate": 1e-5},
            1e-4: {"learning_rate": 1e-4},
            1e-3: {"learning_rate": 1e-3},
            1e-2: {"learning_rate": 1e-2},
        },
    }

    # run once
    comparison_null = {
        "parameter": "norm_type",
        "options": ["layer"],
        "base_changes": {
            "layer": {"norm_type": "layer"},
        },
    }

    print("PRINTING BASE CONFIG")
    print(base_config)
    comparison = comparison_optimizer
    parameter = comparison["parameter"]
    options = comparison["options"]
    base_changes = comparison["base_changes"]

    experiments = []
    for seed in seeds:
        for option in options:
            exp_config = {
                "seed": seed,
                **base_changes[option],
            }
            experiments.append(exp_config)

    final_losses = {option: {} for option in options}

    if n_gpus > 1:
        total_start_time = time.time()
        print(f"\nRunning {len(experiments)} total experiments across {n_gpus} GPUs")
        processes = []
        results_queue = mp.Queue()
        experiments_per_gpu = len(experiments) // n_gpus

        # Track which GPUs are assigned what
        for gpu_id in range(n_gpus):
            start_idx = gpu_id * experiments_per_gpu
            end_idx = (
                start_idx + experiments_per_gpu
                if gpu_id < n_gpus - 1
                else len(experiments)
            )
            gpu_experiments = experiments[start_idx:end_idx]
            print(f"\nGPU {gpu_id} assigned experiments {start_idx} to {end_idx-1}:")
            print(
                f"Experiments: {[(exp[parameter], exp['seed']) for exp in gpu_experiments]}"
            )

            p = mp.Process(
                target=lambda q, gid, exps, param, proj, base: q.put(
                    (gid, run_experiments_on_gpu(gid, exps, param, proj, base))
                ),
                args=(
                    results_queue,
                    gpu_id,
                    gpu_experiments,
                    parameter,
                    project_name,
                    base_config,
                ),
            )
            p.daemon = False
            processes.append(p)
            p.start()

        # Collect results from all processes
        final_losses = {option: {} for option in options}
        print("\nCollecting results from GPUs:")
        for _ in range(len(processes)):
            gpu_id, gpu_results = results_queue.get()  # Unpack the tuple correctly
            print(f"\nReceived results from GPU {gpu_id}:")
            # gpu_results is the dictionary we want to process
            for act_type, seeds_dict in gpu_results.items():  # Process the results dict
                if act_type not in final_losses:
                    final_losses[act_type] = {}
                final_losses[act_type].update(seeds_dict)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        total_elapsed = time.time() - total_start_time
        print(
            f"\nAll experiments completed in {total_elapsed:.2f} seconds"
        )  # Add timing

    elif torch.cuda.is_available():
        # Single GPU setup
        print("Running on single GPU")
        device = torch.device("cuda:0")
        for exp in experiments:
            config = {**base_config, **exp}

            # Create path for CSV logger
            csv_log_path = None
            results_folder = config.get("results_folder", "results")
            csv_log_interval = config.get("csv_log_interval")
            if results_folder and csv_log_interval:
                exp_folder_name = parameter
                option_val_str = str(exp[parameter]).replace("/", "_")
                sub_exp_filename = f"{option_val_str}_seed_{exp['seed']}.csv"
                csv_log_path = os.path.join(
                    results_folder, exp_folder_name, sub_exp_filename
                )

            with wandb.init(project=project_name, config=config):
                train(gpu_id=0, csv_log_path=csv_log_path)
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    else:
        # CPU setup
        print("Running on CPU")
        for exp in experiments:
            config = {**base_config, **exp}

            # Create path for CSV logger
            csv_log_path = None
            results_folder = config.get("results_folder", "results")
            csv_log_interval = config.get("csv_log_interval")
            if results_folder and csv_log_interval:
                exp_folder_name = parameter
                option_val_str = str(exp[parameter]).replace("/", "_")
                sub_exp_filename = f"{option_val_str}_seed_{exp['seed']}.csv"
                csv_log_path = os.path.join(
                    results_folder, exp_folder_name, sub_exp_filename
                )

            with wandb.init(project=project_name, config=config):
                train(csv_log_path=csv_log_path)
                final_loss = wandb.run.summary["val_loss"]
                best_val_loss = wandb.run.summary["best_val_loss"]
                final_losses[exp[parameter]][exp["seed"]] = {
                    "final_loss": final_loss,
                    "best_loss": best_val_loss,
                }

    # Modified CSV output
    csv_data = [[parameter, "Seed", "Final Val Loss", "Best Val Loss"]]
    param_losses = {option: {"final": [], "best": []} for option in options}

    for option in options:
        for seed in seeds:
            result = final_losses[option].get(seed)
            if result is not None:
                final_loss = result["final_loss"]
                best_loss = result["best_loss"]
                param_losses[option]["final"].append(final_loss)
                param_losses[option]["best"].append(best_loss)
                csv_data.append(
                    [
                        option,
                        seed,
                        f"{final_loss:.4f}",
                        f"{best_loss:.4f}",
                    ]
                )

        # Add summary statistics
        if param_losses[option]["final"]:
            mean_final = np.mean(param_losses[option]["final"])
            std_final = np.std(param_losses[option]["final"])
            mean_best = np.mean(param_losses[option]["best"])
            std_best = np.std(param_losses[option]["best"])
            csv_data.append(
                [
                    f"{option}_summary",
                    "N/A",
                    f"{mean_final:.4f} ± {std_final:.4f}",
                    f"{mean_best:.4f} ± {std_best:.4f}",
                ]
            )

    # Save results with parameter type in filename
    csv_file_path = f"experiment_results_{parameter}_{timestamp}.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"\nResults saved to {csv_file_path}")

    # Print summary statistics
    print(f"\nComparing different {parameter} values:")
    for option in options:
        final_loss_values = [
            result["final_loss"] for result in final_losses[option].values()
        ]
        best_loss_values = [
            result["best_loss"] for result in final_losses[option].values()
        ]

        if final_loss_values:
            mean_final = np.mean(final_loss_values)
            std_final = np.std(final_loss_values)
            mean_best = np.mean(best_loss_values)
            std_best = np.std(best_loss_values)
            print(f"{option}:")  # Remove .upper() to preserve exact parameter values
            print(f"  Final: {mean_final:.4f} ± {std_final:.4f}")
            print(f"  Best:  {mean_best:.4f} ± {std_best:.4f}")

    print("\nTOTAL EXPERIMENTS CREATED:")
    print(f"Number of experiments: {len(experiments)}")
    print("Experiments:", [(exp[parameter], exp["seed"]) for exp in experiments])
