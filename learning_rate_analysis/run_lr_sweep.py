"""
Runner for learning rate sweep experiments.

Generates experiment configs from an LR experiment group definition,
slices them for SLURM array parallelism, and runs training via core.train()
for transformers or lstm_training.train_model() for LSTMs.

Usage:
    python run_lr_sweep.py --group modern_transformer --job_id 0 --total_jobs 42
    python run_lr_sweep.py --group lstm_adamw --job_id 0 --total_jobs 42
"""

import argparse
import copy
import math
import os
import sys
import time

# Add parent directory and LSTM_model directory to path
parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "LSTM_model"))

import wandb
import torch
from core import train
from experiment_utils import (
    gen_experim,
    get_base_config,
)
from lstm_experiment_utils import (
    gen_lstm_experim,
    get_lstm_base_config,
)
from lr_experiment_groups import LR_EXPERIMENT_GROUPS, is_combined_group


def build_lr_str(lr):
    """Build a filename-safe string for a learning rate value.

    Emits the base-10 exponent directly: 0.01 -> '-2', 10^-1.25 -> '-1.25'.
    Parser reads any non-'e'-containing token here as log10(lr).
    """
    log_lr = math.log10(lr)
    rounded = round(log_lr, 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{int(round(rounded))}"
    return f"{rounded:g}"


def build_experiments_for_group(group_name):
    """
    Build a flat list of experiment dicts for every (hidden_dim, learning_rate)
    combination in the given group.
    """
    group = LR_EXPERIMENT_GROUPS[group_name]
    hidden_dims = group["hidden_dims"]
    learning_rates = group["learning_rates"]
    model_overrides = group.get("model_overrides", {})
    token_fraction = group.get("token_budget_fraction", 0.05)
    csv_log_interval = group.get("csv_log_interval", 100)
    architecture = group.get("architecture", "transformer")

    results_dir = os.path.join(os.path.dirname(__file__), "results", group_name)

    all_experiments = []

    for dim in hidden_dims:
        if architecture == "lstm":
            base_exp = gen_lstm_experim(dim, label=f"{dim}d", **model_overrides)
            config = get_lstm_base_config()
        else:
            base_exp = gen_experim(dim, label=f"{dim}d", **model_overrides)
            config = get_base_config()

        base_config_overrides = base_exp[0]["subexperiments"][0]
        if "overrides" in base_config_overrides:
            config.update(base_config_overrides["overrides"])
        elif "config" in base_config_overrides:
            config = base_config_overrides["config"]

        # Reduce token budget for the sweep
        full_tokens = config["max_tokens_training"]
        sweep_tokens = max(int(full_tokens * token_fraction), int(129e6 / 8))
        config["max_tokens_training"] = sweep_tokens
        config["csv_log_interval"] = csv_log_interval
        config["results_folder"] = results_dir

        for lr in learning_rates:
            exp_config = copy.deepcopy(config)
            exp_config["learning_rate"] = lr

            label = f"{dim}d_lr_{build_lr_str(lr)}"
            csv_path = os.path.join(results_dir, f"{label}.csv")

            all_experiments.append({
                "config": exp_config,
                "csv_log_path": csv_path,
                "label": label,
                "group_name": group_name,
                "architecture": architecture,
            })

    return all_experiments


def run_experiment(exp, gpu_id):
    """Run a single LR sweep experiment."""
    config = exp["config"]
    label = exp["label"]
    csv_path = exp["csv_log_path"]
    group_name = exp["group_name"]
    architecture = exp.get("architecture", "transformer")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    hidden_key = "hidden_size" if architecture == "lstm" else "hidden_dim"
    print(f"\n{'='*60}")
    print(f"Running: {label}  [{architecture}]")
    print(f"  LR: {config['learning_rate']:.2e}")
    print(f"  {hidden_key}: {config.get(hidden_key)}, layers: {config.get('num_layers')}")
    print(f"  tokens: {config['max_tokens_training']:.2e}")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}")

    start = time.time()
    try:
        if architecture == "lstm":
            from lstm_training import train_model
            # train_model handles wandb.init and wandb.finish internally
            _, results = train_model(config=config, run_name=label, csv_log_path=csv_path)
            elapsed = time.time() - start
            print(f"Completed {label} in {elapsed:.0f}s  "
                  f"val_loss={results['final_val_loss']:.4f}")
        else:
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

    if is_combined_group(args.group):
        print(f"Error: '{args.group}' is a combined (analyze-only) group.")
        print(f"  Submit each of its subgroups individually, e.g. via "
              f"`bash submit_lr.sh <subgroup>`.")
        sys.exit(1)

    group = LR_EXPERIMENT_GROUPS[args.group]
    print(f"\nLR Sweep: {args.group}")
    print(f"  {group['description']}")

    architecture = group.get("architecture", "transformer")
    if architecture != "lstm":
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

    for exp in my_experiments:
        run_experiment(exp, gpu_id)

    print(f"\nJob {args.job_id} complete. Ran {len(my_experiments)} experiments.")


if __name__ == "__main__":
    main()
