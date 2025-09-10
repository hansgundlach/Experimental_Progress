#!/usr/bin/env python3
"""
Test script to print gen_experim output
"""

# Import the function
from experiment_utils import gen_experim, calculate_transformer_params
import json


def main():
    """Print the generated experiment configuration"""

    print("Testing gen_experim function output...")
    print("=" * 60)

    # Generate the experiment
    result = gen_experim(32, label="32d_test_experiment", learning_rate=0.001)

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

    print("\n" + "=" * 60)
    print("Complete Raw Output (JSON format):")
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("Key Scaling Information:")

    # Calculate the number of parameters (with weight tying by default)
    num_params = calculate_transformer_params(
        config["hidden_dim"], config["num_layers"], tie_embeddings=True
    )

    print(f"- Hidden dimension: {config['hidden_dim']}")
    print(f"- Number of layers: {config['num_layers']} (scaled proportionally)")
    print(f"- Number of heads: {config['num_heads']}")
    print(f"- Head dimension: {config['hidden_dim'] // config['num_heads']}")
    print(f"- Total parameters: {num_params:,}")
    print(
        f"- Token limit: {config['token_limit']:,} tokens (20x parameters = {20 * num_params:,})"
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


if __name__ == "__main__":
    main()
