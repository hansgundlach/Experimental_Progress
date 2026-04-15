"""
Runner for learning rate sweep experiments.

Generates experiment configs from an LR experiment group definition,
slices them for SLURM array parallelism, and runs training via core.train().

Usage:
    python run_lr_sweep.py --group modern_transformer --job_id 0 --total_jobs 42
"""

import argparse
import copy
import math
import os
import sys
import time

# Add parent directory so we can import core, experiment_utils, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
import torch
from core import train
from experiment_utils import (
    gen_experim,
    get_base_config,
    calculate_transformer_params,
    estimate_gpu_memory_and_grad_accum,
)
from lr_experiment_groups import LR_EXPERIMENT_GROUPS


def build_experiments_for_group(group_name):
    """
    Build a flat list of (config, csv_log_path, label) tuples for every
    (hidden_dim, learning_rate) combination in the given group.
    """
    group = LR_EXPERIMENT_GROUPS[group_name]
    hidden_dims = group["hidden_dims"]
    learning_rates = group["learning_rates"]
    model_overrides = group.get("model_overrides", {})
    token_fraction = group.get("token_budget_fraction", 0.05)
    csv_log_interval = group.get("csv_log_interval", 100)

    results_dir = os.path.join(os.path.dirname(__file__), "results", group_name)

    all_experiments = []

    for dim in hidden_dims:
        # Use gen_experim to get the properly-scaled base config for this dim
        base_exp = gen_experim(dim, label=f"{dim}d", **model_overrides)
        base_config_overrides = base_exp[0]["subexperiments"][0]

        # Get the full config
        config = get_base_config()
        if "overrides" in base_config_overrides:
            config.update(base_config_overrides["overrides"])
        elif "config" in base_config_overrides:
            config = base_config_overrides["config"]

        # Reduce token budget for the sweep
        full_tokens = config["max_tokens_training"]
        sweep_tokens = max(int(full_tokens * token_fraction), int(129e6 / 8))
        config["max_tokens_training"] = sweep_tokens
        config["csv_log_interval"] = csv_log_interval

        # Set results folder so wandb/logging knows where we are
        config["results_folder"] = results_dir

        for lr in learning_rates:
            exp_config = copy.deepcopy(config)
            exp_config["learning_rate"] = lr

            # Build a filename-safe LR string
            log_lr = math.log10(lr)
            if abs(log_lr - round(log_lr)) < 0.01:
                exponent = int(round(log_lr))
                lr_str = f"10e{exponent:+d}"
            else:
                exponent = math.floor(log_lr)
                coefficient = lr / (10**exponent)
                if abs(coefficient - round(coefficient)) < 0.01:
                    lr_str = f"{round(coefficient):.0f}e{exponent:+d}"
                else:
                    lr_str = f"{coefficient:.1f}e{exponent:+d}"

            label = f"{dim}d_lr_{lr_str}"
            csv_path = os.path.join(results_dir, f"{label}.csv")

            all_experiments.append({
                "config": exp_config,
                "csv_log_path": csv_path,
                "label": label,
                "group_name": group_name,
            })

    return all_experiments


def run_experiment(exp, gpu_id):
    """Run a single LR sweep experiment."""
    config = exp["config"]
    label = exp["label"]
    csv_path = exp["csv_log_path"]
    group_name = exp["group_name"]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  LR: {config['learning_rate']:.2e}")
    print(f"  hidden_dim: {config['hidden_dim']}, layers: {config['num_layers']}")
    print(f"  tokens: {config['max_tokens_training']:.2e}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")

    start = time.time()
    try:
        with wandb.init(
            project=f"lr_sweep_{group_name}",
            config=config,
            name=label,
            reinit=True,
        ) as run:
            results = train(gpu_id=gpu_id, csv_log_path=csv_path)
            elapsed = time.time() - start
            print(f"Completed {label} in {elapsed:.0f}s  "
                  f"val_loss={results['final_val_loss']:.4f}")
            wandb.log({
                "final_val_loss": results["final_val_loss"],
                "final_train_loss": results["final_train_loss"],
                "best_val_loss": results["best_val_loss"],
            })
            return results
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED {label} after {elapsed:.0f}s: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run LR sweep experiments")
    parser.add_argument("--group", required=True,
                        help="Name of the LR experiment group")
    parser.add_argument("--job_id", type=int, default=0,
                        help="SLURM array task ID")
    parser.add_argument("--total_jobs", type=int, default=1,
                        help="Total number of SLURM array tasks")
    args = parser.parse_args()

    if args.group not in LR_EXPERIMENT_GROUPS:
        print(f"Error: Unknown group '{args.group}'")
        print(f"Available groups: {list(LR_EXPERIMENT_GROUPS.keys())}")
        sys.exit(1)

    group = LR_EXPERIMENT_GROUPS[args.group]
    print(f"\nLR Sweep: {args.group}")
    print(f"  {group['description']}")

    wandb.login()

    # Build all experiments for this group
    all_experiments = build_experiments_for_group(args.group)
    total_count = len(all_experiments)
    print(f"  Total experiments in group: {total_count}")

    # Slice for this SLURM task
    exps_per_job = (total_count + args.total_jobs - 1) // args.total_jobs
    start_idx = args.job_id * exps_per_job
    end_idx = min(start_idx + exps_per_job, total_count)
    my_experiments = all_experiments[start_idx:end_idx]

    print(f"  This job ({args.job_id}/{args.total_jobs}): "
          f"experiments {start_idx}-{end_idx-1} ({len(my_experiments)} total)")

    # Detect GPU
    n_gpus = torch.cuda.device_count()
    gpu_id = 0 if n_gpus > 0 else None
    print(f"  GPU: {'cuda:0' if gpu_id is not None else 'CPU'}")

    # Run experiments sequentially on this GPU
    for exp in my_experiments:
        run_experiment(exp, gpu_id)

    print(f"\nJob {args.job_id} complete. Ran {len(my_experiments)} experiments.")


if __name__ == "__main__":
    main()
