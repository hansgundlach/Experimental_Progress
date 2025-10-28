#!/usr/bin/env python3
"""
Test script to print gen_experim output
"""

# Import the function
from experiment_utils import (
    gen_experim,
    calculate_transformer_params,
    calculate_non_embedding_params,
)
import json


def analyze_experiment(hidden_dim, label_suffix=""):
    """Analyze a single experiment configuration"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT ANALYSIS: {hidden_dim}d Model{label_suffix}")
    print("=" * 60)

    # Generate the experiment
    result = gen_experim(
        hidden_dim, label=f"{hidden_dim}d_test_experiment", learning_rate=0.001
    )

    # Extract the configuration
    exp_group = result[0]
    sub_exp = exp_group["subexperiments"][0]
    config = sub_exp["overrides"]

    print(f"Experiment Group Name: {exp_group['name']}")
    print(f"Label: {sub_exp['label']}")
    print("\nGenerated Configuration Parameters:")
    print("-" * 40)

    # Print each parameter nicely formatted
    for key, value in sorted(config.items()):
        if isinstance(value, (int, float)) and value > 1000:
            print(f"{key:25}: {value:,}")
        else:
            print(f"{key:25}: {value}")

    print("\n" + "-" * 40)
    print("Key Scaling Information:")

    # Calculate the number of parameters (with weight tying by default)
    num_params = calculate_transformer_params(
        config["hidden_dim"], config["num_layers"], tie_embeddings=True
    )

    # Calculate non-embedding parameters (Kaplan et al. definition)
    non_embedding_params = calculate_non_embedding_params(
        config["hidden_dim"], config["num_layers"], tie_embeddings=True
    )

    print(f"- Hidden dimension: {config['hidden_dim']}")
    print(f"- Number of layers: {config['num_layers']} (scaled proportionally)")
    print(f"- Number of heads: {config['num_heads']}")
    print(f"- Head dimension: {config['hidden_dim'] // config['num_heads']}")
    print(f"- Total parameters: {num_params:,}")
    print(
        f"- Non-embedding parameters: {non_embedding_params:,} (Kaplan et al. definition)"
    )
    print(
        f"- Token limit: {config['max_tokens_training']:,} tokens (20x parameters = {20 * num_params:,})"
    )
    print(f"- Gradient accumulation: {config['gradient_accumulation_steps']} steps")

    # Calculate batch size breakdown
    from experiment_utils import get_base_config

    base_config = get_base_config()
    target_effective_batch_size = base_config["target_effective_batch_size"]
    per_step_batch_size = config["batch_size"]  # Calculated per-step batch size
    grad_accum = config["gradient_accumulation_steps"]
    effective_batch_size = per_step_batch_size * grad_accum

    print(
        f"- Target effective batch size: {target_effective_batch_size} (optimization goal)"
    )
    print(
        f"- Per-step batch size: {per_step_batch_size} (calculated to fit in GPU memory)"
    )
    print(f"- Gradient accumulation steps: {grad_accum}")
    print(f"- Effective batch size: {effective_batch_size} (per_step × grad_accum)")
    print(f"- Learning rate: {config['learning_rate']} (your override)")


def main():
    """Print the generated experiment configurations for multiple hidden dimensions"""

    print("Testing gen_experim function output for multiple model sizes...")

    # Test different hidden dimensions
    hidden_dims = [32, 48, 64, 96, 128, 160, 256]

    for i, hidden_dim in enumerate(hidden_dims):
        analyze_experiment(hidden_dim, f" ({i+1}/4)")

    print(f"\n{'='*60}")
    print("SUMMARY: All experiments analyzed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# test git
