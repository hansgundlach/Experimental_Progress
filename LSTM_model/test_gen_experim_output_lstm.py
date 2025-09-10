#!/usr/bin/env python3
"""
Test script to print gen_lstm_experim output
"""

# Import the function
from lstm_experiment_utils import gen_lstm_experim, calculate_lstm_params
import json


def main():
    """Print the generated LSTM experiment configuration"""

    print("Testing gen_lstm_experim function output...")
    print("=" * 60)

    # Test different hidden sizes
    test_hidden_sizes = [16, 32, 48, 64]
    
    for hidden_size in test_hidden_sizes:
        print(f"\n--- Testing hidden_size = {hidden_size} ---")
        
        # Generate the experiment
        result = gen_lstm_experim(hidden_size, label=f"lstm_{hidden_size}d_test_experiment", learning_rate=0.01)

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

        print("\nKey Scaling Information:")
        print("-" * 30)

        # Calculate the number of parameters (with weight tying by default)
        param_info = calculate_lstm_params(
            config["hidden_size"], config["num_layers"], tie_embeddings=True
        )

        print(f"- Hidden size: {config['hidden_size']}")
        print(f"- Number of layers: {config['num_layers']} (scaled proportionally)")
        print(f"- Total parameters: {param_info['total_params']:,}")
        print(f"- Trainable parameters: {param_info['trainable_params']:,}")
        print(f"  - Embedding params: {param_info['embedding_params']:,}")
        print(f"  - LSTM params: {param_info['lstm_params']:,}")
        print(f"  - Final layer params: {param_info['final_layer_params']:,}")
        print(
            f"- Character limit: {int(config['max_characters']):,} chars (20x trainable params)"
        )
        print(f"- Token estimate: {int(config['max_characters']) // 4:,} tokens (4:1 char:token ratio)")
        print(f"- Gradient accumulation: {config['gradient_accumulation_steps']} steps")
        print(f"- Learning rate: {config.get('learning_rate', 'default')} (your override)")

    print("\n" + "=" * 60)
    print("Complete Raw Output Example (JSON format for 32d):")
    result_32d = gen_lstm_experim(32, label="lstm_32d_example")
    print(json.dumps(result_32d, indent=2))

    print("\n" + "=" * 60)
    print("Parameter Calculation Verification:")
    print("-" * 40)
    
    # Test parameter calculation directly
    for hidden_size in [16, 32, 64]:
        param_info = calculate_lstm_params(hidden_size, num_layers=2, tie_embeddings=True)
        print(f"\nHidden size {hidden_size}:")
        print(f"  Formula verification:")
        print(f"  - Embedding: 50257 * {hidden_size} = {50257 * hidden_size:,}")
        print(f"  - LSTM (per layer): 8 * {hidden_size}^2 + 4 * {hidden_size} = {8 * hidden_size**2 + 4 * hidden_size:,}")
        print(f"  - LSTM (2 layers): {2 * (8 * hidden_size**2 + 4 * hidden_size):,}")
        print(f"  - Final bias: 50257")
        expected_trainable = 50257 * hidden_size + 2 * (8 * hidden_size**2 + 4 * hidden_size) + 50257
        print(f"  - Expected trainable: {expected_trainable:,}")
        print(f"  - Calculated trainable: {param_info['trainable_params']:,}")
        print(f"  - Match: {'✓' if expected_trainable == param_info['trainable_params'] else '✗'}")

    print("\n" + "=" * 60)
    print("Memory Estimation Test:")
    print("-" * 30)
    
    from lstm_experiment_utils import estimate_lstm_gpu_memory_and_grad_accum
    
    # Test memory estimation for different configurations
    test_configs = [
        (16, 2, "V100"),
        (32, 2, "V100"), 
        (64, 4, "V100"),
        (32, 2, "H100"),
        (64, 4, "H100"),
    ]
    
    for hidden_size, num_layers, gpu_type in test_configs:
        grad_accum = estimate_lstm_gpu_memory_and_grad_accum(
            hidden_size, num_layers, batch_size=32, seq_length=128, gpu_type=gpu_type, world_size=2
        )
        param_info = calculate_lstm_params(hidden_size, num_layers, tie_embeddings=True)
        print(f"{hidden_size}d, {num_layers} layers, {gpu_type}: {grad_accum} grad_accum steps ({param_info['trainable_params']:,} params)")


if __name__ == "__main__":
    main()