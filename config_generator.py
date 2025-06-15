import math
import copy


def get_optimal_config_for_compute(compute_budget, base_config=None):
    """
    Intelligently scale configuration parameters based on compute budget using best practices.

    Args:
        compute_budget (float): Relative compute multiplier (1.0 = baseline ~1M parameters)
        base_config (dict, optional): Base configuration to modify. If None, uses sensible defaults.

    Returns:
        dict: Optimized configuration scaled for the compute budget
    """

    # Default minimal base config if none provided
    if base_config is None:
        base_config = {
            "dataset": "wikitext",
            "pos_encoding": "sinusoidal",
            "init_scheme": "xavier_uniform",
            "stride": None,  # Will be set based on seq_length
            "pin_memory": True,
            "compile": False,
            "prefetch_factor": 4,
            "use_gradient_clipping": True,
            "gradient_clip_val": 1.0,
            "label_smoothing": 0.0,
            "optimizer": "adamw",
            "activation": "relu",
            "norm_type": "layer",
            "min_lr": 1e-5,
            "lr_schedule": "cosine",
            "warmup_epochs_frac": 0.1,
            "weight_decay": 0.1,
            "dropout": 0.0,
        }

    config = copy.deepcopy(base_config)

    # 1. Scale model architecture based on compute budget
    # Use approximately: parameters ~ compute_budget^0.5 (following scaling laws)
    param_scale = compute_budget**0.5

    # Base model size (targeting ~1M params at compute_budget=1.0)
    base_hidden_dim = 32
    base_num_layers = 3

    # Scale hidden dimension (primary driver of parameter count)
    config["hidden_dim"] = max(8, int(base_hidden_dim * param_scale))

    # Ensure hidden_dim is divisible by num_heads for efficiency
    # Scale num_heads with model size but keep head_dim reasonable (32-128)
    target_head_dim = 64
    config["num_heads"] = max(
        1, min(config["hidden_dim"] // 16, config["hidden_dim"] // target_head_dim)
    )

    # Adjust hidden_dim to be divisible by num_heads
    config["hidden_dim"] = config["num_heads"] * (
        config["hidden_dim"] // config["num_heads"]
    )

    # Scale depth more conservatively (depth harder to optimize)
    config["num_layers"] = max(1, int(base_num_layers * (compute_budget**0.3)))

    # 2. Scale batch size with compute budget (larger models benefit from larger batches)
    base_batch_size = 64
    config["batch_size"] = max(1, int(base_batch_size * (compute_budget**0.4)))

    # 3. Scale learning rate with batch size (sqrt scaling rule)
    base_lr = 1e-3
    batch_scale = config["batch_size"] / base_batch_size
    config["learning_rate"] = base_lr * math.sqrt(batch_scale)

    # 4. Scale sequence length with compute budget (longer sequences need more compute)
    base_seq_length = 128
    config["seq_length"] = max(32, int(base_seq_length * (compute_budget**0.25)))

    # Set stride to 50% overlap for efficiency
    config["stride"] = config["seq_length"] // 2

    # 5. Scale dataset size with compute budget (more compute = can handle more data)
    base_data_limit = 1e7
    config["wikitext_limit"] = int(base_data_limit * compute_budget)

    # 6. Scale training epochs with compute budget (more compute = longer training)
    base_epochs = 3
    config["max_epochs"] = max(1, int(base_epochs * (compute_budget**0.3)))
    config["min_epochs"] = config["max_epochs"]

    # 7. Scale gradient accumulation for memory efficiency with large models
    if compute_budget >= 4.0:
        config["gradient_accumulation_steps"] = max(4, int(8 * (compute_budget**0.2)))
    else:
        config["gradient_accumulation_steps"] = max(1, int(2 * compute_budget))

    # 8. Use better techniques with more compute
    if compute_budget >= 2.0:
        config["activation"] = "gelu"  # Better than relu for larger models
        config["lr_schedule"] = "cosine_warmup"  # Better scheduling
        config["prefetch_factor"] = 8

    if compute_budget >= 4.0:
        config["pos_encoding"] = "rotary"  # Better positional encoding
        config["init_scheme"] = "transformer_scaled"  # Better initialization
        config["activation"] = "swiglu"  # Even better activation (but more compute)

    if compute_budget >= 8.0:
        config["compile"] = True  # Torch compilation for speed
        config["norm_type"] = "rms"  # Slightly more efficient

    # 9. Adjust prefetch factor based on model complexity
    if compute_budget <= 0.5:
        config["prefetch_factor"] = 2
    elif compute_budget >= 8.0:
        config["prefetch_factor"] = 16

    # 10. Memory management for different compute scales
    if compute_budget < 1.0:
        # Small models - prioritize speed
        config["pin_memory"] = False
        config["compile"] = False
    elif compute_budget >= 16.0:
        # Large models - aggressive memory optimization
        config["pin_memory"] = True
        config["compile"] = True

    return config


def estimate_model_parameters(config):
    """
    Estimate the number of parameters in the model given a config.

    Args:
        config (dict): Model configuration

    Returns:
        int: Estimated number of parameters
    """
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    vocab_size = 50257  # GPT-2 tokenizer size
    seq_length = config["seq_length"]

    # Embedding layer
    embedding_params = vocab_size * hidden_dim

    # Positional encoding
    if config["pos_encoding"] == "learned":
        pos_params = seq_length * hidden_dim
    else:
        pos_params = 0  # Sinusoidal and rotary don't add parameters

    # Each transformer layer
    # Attention: Q, K, V projections + output projection
    attn_params = 4 * (hidden_dim * hidden_dim)

    # Feed forward
    if config["activation"] in ["swiglu", "glu"]:
        # GLU variants use 2x hidden dim in intermediate layer
        ff_params = 2 * (hidden_dim * (hidden_dim * 4)) + (hidden_dim * 4) * hidden_dim
    else:
        # Standard FF: up projection + down projection
        ff_params = (hidden_dim * (hidden_dim * 4)) + ((hidden_dim * 4) * hidden_dim)

    # Layer norm parameters (minimal)
    norm_params = 2 * hidden_dim  # weight + bias

    layer_params = attn_params + ff_params + norm_params
    total_layer_params = num_layers * layer_params

    # Final output layer
    output_params = hidden_dim * vocab_size

    total_params = embedding_params + pos_params + total_layer_params + output_params

    return total_params


def estimate_training_flops(config):
    """
    Estimate training FLOPs based on model size and training setup.

    Args:
        config (dict): Model configuration

    Returns:
        float: Estimated total training FLOPs
    """
    params = estimate_model_parameters(config)
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    max_epochs = config["max_epochs"]
    wikitext_limit = config["wikitext_limit"]

    # Estimate number of tokens
    chars_per_token = 4  # Rough estimate for English text
    total_tokens = wikitext_limit // chars_per_token
    tokens_per_epoch = total_tokens

    # Estimate steps per epoch
    tokens_per_batch = batch_size * seq_length
    steps_per_epoch = tokens_per_epoch // tokens_per_batch
    total_steps = steps_per_epoch * max_epochs

    # FLOPs per forward pass ≈ 2 * params * tokens_per_batch
    # Factor of 3 for backward pass (rough estimate)
    flops_per_step = 6 * params * tokens_per_batch
    total_flops = flops_per_step * total_steps

    return total_flops


# ====================================================================
# convert parameters to config

import math
import copy


def get_optimal_config_for_params(target_params_str, base_config=None):
    """
    Generates an optimal transformer configuration for a target parameter count
    using functional scaling laws and industry best practices.

    Args:
        target_params_str (str): The target parameter count (e.g., "1M", "10M", "100M").
        base_config (dict, optional): A base configuration to build upon.
                                      If None, sensible defaults are used.

    Returns:
        dict: The optimized configuration.
    """
    target_params = parse_param_string(target_params_str)

    # --- 1. Set sensible defaults for the base config ---
    if base_config is None:
        base_config = {
            "dataset": "wikitext",
            "pin_memory": True,
            "use_gradient_clipping": True,
            "gradient_clip_val": 1.0,
            "label_smoothing": 0.0,
            "optimizer": "adamw",
            "weight_decay": 0.1,
            "dropout": 0.0,
        }
    config = copy.deepcopy(base_config)

    # --- 2. Determine architecture and features based on model scale ---
    # For larger models, more advanced components are justified.
    config["activation"] = "swiglu" if target_params > 10e6 else "gelu"
    config["norm_type"] = "rms" if target_params > 50e6 else "layer"
    config["pos_encoding"] = "rotary" if target_params > 10e6 else "sinusoidal"
    config["init_scheme"] = (
        "transformer_scaled" if target_params > 10e6 else "xavier_uniform"
    )
    config["compile"] = True if target_params > 50e6 else False

    # --- 3. Find an optimal architecture for the target parameter count ---
    # This is a search problem to find the best width/depth trade-off.
    hidden_dim, num_layers, num_heads = find_optimal_architecture(
        target_params, config["activation"]
    )
    config.update(
        {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }
    )

    # --- 4. Apply functional scaling laws for other hyperparameters ---

    # Data Scaling (Chinchilla's 20x rule: 20 tokens per parameter)
    # Assuming ~4 characters per token for English text.
    chars_per_token = 4
    config["wikitext_limit"] = int(target_params * 20 * chars_per_token)

    # Batch Size Scaling (Gopher paper suggests scaling with model size)
    # We use a gentle power law, clamped to a reasonable range.
    base_params = 1e6  # 1M
    base_batch_size = 128
    # Batch size scales with params^(1/4)
    batch_size_scale_factor = (target_params / base_params) ** 0.25
    config["batch_size"] = int(min(2048, base_batch_size * batch_size_scale_factor))

    # Learning Rate Scaling (Larger models often use smaller LRs)
    # We use an inverse power law for the peak learning rate.
    base_lr = 3e-4
    # LR scales with params^(-1/5)
    lr_scale_factor = (target_params / base_params) ** -0.2
    config["learning_rate"] = base_lr * lr_scale_factor
    config["min_lr"] = config["learning_rate"] / 10

    # LR Schedule: Use more sophisticated schedules for larger models
    config["lr_schedule"] = "cosine_warmup" if target_params > 1e6 else "cosine"
    config["warmup_epochs_frac"] = 0.1

    # Sequence Length Scaling
    # Scales slowly with model size, as attention cost is quadratic.
    base_seq_len = 256
    # Scales with log of parameter count
    seq_len_scale_factor = math.log(target_params / base_params + 1) / math.log(4)
    config["seq_length"] = int(base_seq_len * (1 + seq_len_scale_factor))
    config["seq_length"] = min(config["seq_length"], 4096)
    config["stride"] = config["seq_length"] // 2

    # Training Duration (Epochs)
    # With a Chinchilla-optimal dataset, we need fewer epochs.
    # We can train for longer on smaller models to compensate for less data.
    config["max_epochs"] = int(max(3, 20 - 2 * math.log10(target_params)))
    config["min_epochs"] = config["max_epochs"]

    # Gradient Accumulation (for memory management)
    # Increase accumulation for larger models to keep per-device batch size small.
    config["gradient_accumulation_steps"] = max(1, config["batch_size"] // 128)

    # Dataloader optimization
    config["prefetch_factor"] = 8 if target_params > 1e6 else 4

    return config


def find_optimal_architecture(target_params, activation, vocab_size=50257):
    """Finds hidden_dim, num_layers, and num_heads for a target parameter count."""
    best_config = None
    best_diff = float("inf")

    # Heuristic: hidden_dim grows roughly as sqrt(params)
    min_hidden = 32
    max_hidden = int(4096 * (target_params / 1e8) ** 0.5)  # Scale max search space

    # Iterate through reasonable hidden dimensions (must be divisible by 16)
    for hidden_dim in range(min_hidden, max_hidden + 1, 16):
        # Prefer head dimensions of 64 or 128 for efficiency on modern GPUs
        for head_dim in [64, 128]:
            if hidden_dim % head_dim != 0:
                continue
            num_heads = hidden_dim // head_dim

            # From the equation `params = L * (layer_params) + other_params`,
            # we can directly solve for num_layers, L.
            embedding_params = vocab_size * hidden_dim
            output_params = hidden_dim * vocab_size
            non_layer_params = embedding_params + output_params

            layer_params = calculate_layer_params(hidden_dim, activation)

            # Solve for L
            required_layers = (target_params - non_layer_params) / layer_params
            num_layers = max(1, round(required_layers))

            params = non_layer_params + num_layers * layer_params
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_config = (hidden_dim, num_layers, num_heads)

    if best_config is None:
        raise ValueError(
            f"Could not find a suitable architecture for {target_params:,} parameters."
        )

    return best_config


def calculate_layer_params(hidden_dim, activation):
    """Calculates the parameter count for a single transformer layer."""
    # Attention block (QKV + Output projection)
    attn_params = 4 * (hidden_dim * hidden_dim)

    # Feed-forward block
    if activation == "swiglu":
        # SwiGLU uses a wider intermediate layer but is gated. A common setup
        # is to make it have roughly the same params as a standard FFN.
        # Here we use the implementation from the user's `core.py`
        ff_hidden_dim = hidden_dim * 4
        # proj (dim -> hidden*2) and to_out (hidden -> dim)
        ff_params = (hidden_dim * ff_hidden_dim * 2) + (ff_hidden_dim * hidden_dim)
    else:  # GELU, ReLU
        ff_hidden_dim = hidden_dim * 4
        # Standard FFN (up-proj and down-proj)
        ff_params = (hidden_dim * ff_hidden_dim) + (ff_hidden_dim * hidden_dim)

    # LayerNorms (one before attention, one before FFN)
    norm_params = 2 * (2 * hidden_dim)  # 2 norms, each with weight+bias

    # We ignore biases for simplicity in this calculation, as they are a small fraction.
    return attn_params + ff_params + norm_params


def parse_param_string(s):
    """Parses strings like '1M', '10.5B' into integers."""
    s = s.upper().strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    return int(s)


# Example Usage:


# Example usage and testing
if __name__ == "__main__":
    print("Compute Budget Scaling Examples:")
    print("=" * 50)

    compute_levels = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    for compute_budget in compute_levels:
        config = get_optimal_config_for_compute(compute_budget)
        params = estimate_model_parameters(config)
        flops = estimate_training_flops(config)

        print(f"\nCompute Budget: {compute_budget:.1f}x")
        print(
            f"  Model: {config['hidden_dim']}d, {config['num_layers']}L, {config['num_heads']}H"
        )
        print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        print(
            f"  Batch Size: {config['batch_size']}, LR: {config['learning_rate']:.2e}"
        )
        print(f"  Seq Length: {config['seq_length']}, Epochs: {config['max_epochs']}")
        print(f"  Est. FLOPs: {flops:.2e}")
        print(
            f"  Activation: {config['activation']}, Pos Enc: {config['pos_encoding']}"
        )

    print("Functionally-Derived Configurations:")
    print("=" * 60)

    target_sizes = ["100K", "1M", "10M", "70M", "160M", "1B"]

    for target in target_sizes:
        config = get_optimal_config_for_params(target)

        # Estimate final params for reporting
        layer_p = calculate_layer_params(config["hidden_dim"], config["activation"])
        final_params = 50257 * config["hidden_dim"] * 2 + config["num_layers"] * layer_p

        print(f"\nTarget: ~{target} params")
        print(f"  --> Actual: ~{final_params/1e6:.2f}M params")
        print(
            f"  Architecture: {config['hidden_dim']}d | {config['num_layers']}L | {config['num_heads']}H"
        )
        print(
            f"  Training: {config['batch_size']} batch | {config['learning_rate']:.2e} LR | {config['max_epochs']} epochs"
        )
        print(
            f"  Data: {config['seq_length']} seq_len | {config['wikitext_limit']/1e6:.1f}M chars limit"
        )
        print(
            f"  Features: {config['activation']} | {config['norm_type']} | {config['pos_encoding']}"
        )
