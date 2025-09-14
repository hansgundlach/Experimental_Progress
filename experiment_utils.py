# experiment_utils.py
import copy
import math


def create_multi_seed_experiments(base_experiments, seeds):
    """
    Create multiple versions of experiments with different random seeds.

    Args:
        base_experiments: List of experiment dictionaries (e.g., HIDDEN_DIM_EXPERIMENTS)
        seeds: List of seed values (e.g., [123, 789])

    Returns:
        List of experiment dictionaries with seed variations
    """
    multi_seed_experiments = []

    for experiment in base_experiments:
        # Create a new experiment group for each base experiment
        new_experiment = {"name": experiment["name"], "subexperiments": []}

        # For each subexperiment in the base experiment
        for sub_exp in experiment["subexperiments"]:
            # Create a version for each seed
            for seed in seeds:
                # Create new subexperiment with seed suffix
                new_sub_exp = copy.deepcopy(sub_exp)

                # Add seed to the label
                original_label = sub_exp["label"]
                new_sub_exp["label"] = f"{original_label}_{seed}"

                # Add seed to overrides (or config if using pre-generated configs)
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["seed"] = seed
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["seed"] = seed
                else:
                    # If neither exists, create overrides with just the seed
                    new_sub_exp["overrides"] = {"seed": seed}

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_seed_experiments.append(new_experiment)

    return multi_seed_experiments


def create_multi_lr_experiments(base_experiments, learning_rates):
    """
    Create multiple versions of experiments with different learning rates.
    Similar to create_multi_seed_experiments but for learning rates.

    Args:
        base_experiments: List of experiment dictionaries (e.g., LSTM_HIDDEN_DIM_EXPERIMENTS)
        learning_rates: List of learning rate values (e.g., [1e-4, 1e-3, 1e-2])

    Returns:
        List of experiment dictionaries with learning rate variations
    """
    multi_lr_experiments = []

    for experiment in base_experiments:
        # Check if any sub-experiment has a custom results folder
        custom_folder = None
        for sub_exp in experiment["subexperiments"]:
            if "overrides" in sub_exp:
                # Check for both folder_name and results_folder
                custom_folder = sub_exp["overrides"].get("folder_name") or sub_exp[
                    "overrides"
                ].get("results_folder")
                if custom_folder:
                    break
            elif "config" in sub_exp:
                custom_folder = sub_exp["config"].get("folder_name") or sub_exp[
                    "config"
                ].get("results_folder")
                if custom_folder:
                    break

        # Create a new experiment group name
        if custom_folder:
            # If there's a custom folder, create the name to put results in "custom_folder_lr_sweep/"
            new_experiment_name = f"{custom_folder}_lr_sweep"
        else:
            # Otherwise use the original experiment name with lr_sweep suffix
            new_experiment_name = f"{experiment['name']}_lr_sweep"

        new_experiment = {
            "name": new_experiment_name,
            "subexperiments": [],
        }

        # For each subexperiment in the base experiment
        for sub_exp in experiment["subexperiments"]:
            # Create a version for each learning rate
            for lr in learning_rates:
                # Create new subexperiment with lr suffix
                new_sub_exp = copy.deepcopy(sub_exp)

                # Add learning rate to the label
                original_label = sub_exp["label"]
                # Format learning rate for filename-safe label with clear scientific notation
                import math

                if lr >= 1:
                    lr_str = f"{lr:.0f}"
                else:
                    # Use clear scientific notation: 10e-1, 10e-2, 10e-3, etc.
                    log_lr = math.log10(lr)

                    # Check if it's close to a nice power of 10
                    if (
                        abs(log_lr - round(log_lr)) < 0.01
                    ):  # Very close to integer power
                        exponent = int(round(log_lr))
                        lr_str = f"10e{exponent:+d}"  # +d ensures +/- sign
                    else:
                        # For non-integer powers, use coefficient notation
                        exponent = math.floor(log_lr)
                        coefficient = lr / (10**exponent)
                        if abs(coefficient - round(coefficient)) < 0.01:
                            lr_str = f"{round(coefficient):.0f}e{exponent:+d}"
                        else:
                            lr_str = f"{coefficient:.1f}e{exponent:+d}"

                new_sub_exp["label"] = f"{original_label}_lr_{lr_str}"

                # Add learning rate and token_limit to overrides, and update folder settings
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["learning_rate"] = lr
                    new_sub_exp["overrides"]["token_limit"] = int(
                        129e6 / 4
                    )  # Convert from old char estimate
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["overrides"].pop("folder_name", None)
                        new_sub_exp["overrides"].pop("results_folder", None)
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["token_limit"] = int(
                        129e6 / 4
                    )  # Convert from old char estimate
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["config"].pop("folder_name", None)
                        new_sub_exp["config"].pop("results_folder", None)
                else:
                    # If neither exists, create overrides with learning rate and token_limit
                    overrides_dict = {
                        "learning_rate": lr,
                        "token_limit": int(129e6 / 4),  # Convert from old char estimate
                    }
                    # Don't add custom folder for new overrides - let experiment name handle it
                    new_sub_exp["overrides"] = overrides_dict

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    return multi_lr_experiments


def calculate_transformer_params(
    hidden_dim,
    num_layers,
    vocab_size=50257,
    seq_length=128,
    pos_encoding="rotary",
    tie_embeddings=True,
):
    """
    Calculate the total number of parameters for a transformer model.

    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size (default GPT-2 vocab size)
        seq_length: Sequence length (for learned positional embeddings)
        pos_encoding: Type of positional encoding ("rotary", "learned", "sinusoidal")
        tie_embeddings: Whether input and output embeddings are tied (default True)

    Returns:
        Total number of parameters
    """
    # Embedding parameters: vocab_size * hidden_dim
    embedding_params = vocab_size * hidden_dim

    # Positional embedding parameters
    pos_emb_params = 0
    if pos_encoding == "learned":
        pos_emb_params = seq_length * hidden_dim
    # rotary and sinusoidal don't add parameters

    # Per-layer parameters (based on standard transformer architecture)
    # Each layer has:
    # - Multi-head attention: 4 * hidden_dim^2 (Q, K, V, O projections)
    # - Feed-forward: 2 * hidden_dim * (4 * hidden_dim) = 8 * hidden_dim^2
    # - Layer norms: 2 * hidden_dim (small, can be ignored for scaling)
    params_per_layer = 4 * hidden_dim**2 + 8 * hidden_dim**2  # = 12 * hidden_dim^2
    layer_params = num_layers * params_per_layer

    # Final linear layer: hidden_dim * vocab_size (only if not tied to embedding)
    final_layer_params = 0 if tie_embeddings else hidden_dim * vocab_size

    # Final layer bias: vocab_size (always present even with weight tying)
    final_bias_params = vocab_size

    total_params = (
        embedding_params
        + pos_emb_params
        + layer_params
        + final_layer_params
        + final_bias_params
    )
    return total_params


def calculate_non_embedding_params(
    hidden_dim,
    num_layers,
    vocab_size=50257,
    seq_length=128,
    pos_encoding="rotary",
    tie_embeddings=True,
):
    """
    Calculate the number of non-embedding parameters for a transformer model.
    This follows Kaplan et al. (2020) definition where non-embedding parameters
    exclude input embeddings, output embeddings (if tied), and positional embeddings.

    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size (default GPT-2 vocab size)
        seq_length: Sequence length (for learned positional embeddings)
        pos_encoding: Type of positional encoding ("rotary", "learned", "sinusoidal")
        tie_embeddings: Whether input and output embeddings are tied (default True)

    Returns:
        Number of non-embedding parameters
    """
    # Per-layer parameters (based on standard transformer architecture)
    # Each layer has:
    # - Multi-head attention: 4 * hidden_dim^2 (Q, K, V, O projections)
    # - Feed-forward: 2 * hidden_dim * (4 * hidden_dim) = 8 * hidden_dim^2
    # - Layer norms: 2 * hidden_dim (small, can be ignored for scaling)
    params_per_layer = 4 * hidden_dim**2 + 8 * hidden_dim**2  # = 12 * hidden_dim^2
    layer_params = num_layers * params_per_layer

    # Final linear layer: hidden_dim * vocab_size (only if not tied to embedding)
    final_layer_params = 0 if tie_embeddings else hidden_dim * vocab_size

    # Final layer bias: vocab_size (always present even with weight tying)
    final_bias_params = vocab_size

    # Non-embedding parameters = layer params + final layer + final bias
    # Note: We exclude input embeddings, output embeddings (if tied), and positional embeddings
    non_embedding_params = layer_params + final_layer_params + final_bias_params

    return non_embedding_params


def estimate_gpu_memory_and_grad_accum(
    hidden_dim, num_layers, target_effective_batch_size, seq_length, gpu_type="V100"
):
    """
    Estimate optimal per-step batch size and gradient accumulation steps.

    FIXED VERSION: Now properly calculates per-step batch size to avoid OOM, then uses
    gradient accumulation to reach target effective batch size.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        target_effective_batch_size: Target effective batch size (what you want for training)
        seq_length: Sequence length
        gpu_type: "V100" (32GB) or "H100" (80GB)

    Returns:
        Suggested gradient accumulation steps (effective = per_step * grad_accum)
    """
    # GPU memory limits (based on actual OOM error showing 31.73 GiB total capacity)
    gpu_memory = {
        "V100": 30 * 1024**3,  # ~30GB usable out of 32GB
        "H100": 76 * 1024**3,  # ~76GB usable out of 80GB
    }

    vocab_size = 50257  # GPT-2 vocab size
    available_memory = gpu_memory.get(gpu_type, gpu_memory["V100"])

    # Use more conservative memory thresholds for larger models
    # Larger models have much more complex memory patterns and spikes
    if hidden_dim >= 128:
        memory_threshold = 0.35 * available_memory  # Very conservative for large models
    elif hidden_dim >= 64:
        memory_threshold = 0.45 * available_memory  # Conservative for medium models
    else:
        memory_threshold = (
            0.60 * available_memory
        )  # Original threshold for small models

    # Model parameters (constant regardless of batch size)
    params = calculate_transformer_params(hidden_dim, num_layers, tie_embeddings=True)
    model_memory = params * 4  # fp32 model weights
    optimizer_memory = params * 4 * 4  # Adam: weights + gradients + 2 momentum terms

    # Calculate number of heads (same logic as gen_experim)
    target_head_dim = 32
    num_heads = max(1, int(round(hidden_dim / target_head_dim)))
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    # Fixed memory (independent of batch size)
    fixed_memory = model_memory + optimizer_memory

    # Memory available for activations (batch-dependent)
    available_for_activations = memory_threshold - fixed_memory

    if available_for_activations <= 0:
        # Model itself is too big, use minimum settings
        return 64  # Maximum gradient accumulation

    # Calculate memory per batch-size unit for different components
    # All calculations are per single batch item [1, seq_length, ...]

    # 1. CRITICAL: Final layer (logits) tensor - usually the biggest!
    # [batch_size, seq_length, vocab_size] + gradients
    final_layer_per_batch = 2 * seq_length * vocab_size * 4  # output + gradients

    # 2. Forward pass activations
    forward_activations_per_batch = seq_length * hidden_dim * num_layers * 4

    # 3. Attention matrices: [batch, heads, seq_len, seq_len] (scores + weights)
    attention_per_batch = 2 * num_heads * seq_length * seq_length * 4

    # 4. QKV and feed-forward
    qkv_per_batch = 3 * seq_length * hidden_dim * 4
    ff_per_batch = seq_length * (4 * hidden_dim) * num_layers * 4

    # 5. Gradient storage for all activations
    gradient_storage_per_batch = (
        forward_activations_per_batch
        + attention_per_batch
        + qkv_per_batch
        + ff_per_batch
    )

    # Total memory per batch item
    memory_per_batch_item = (
        final_layer_per_batch
        + forward_activations_per_batch
        + attention_per_batch
        + qkv_per_batch
        + ff_per_batch
        + gradient_storage_per_batch
    )

    # Add overhead for memory fragmentation and misc tensors (more for larger models)
    if hidden_dim >= 128:
        memory_per_batch_item *= (
            1.5  # 50% overhead for large models (complex activation patterns)
        )
    elif hidden_dim >= 64:
        memory_per_batch_item *= 1.3  # 30% overhead for medium models
    else:
        memory_per_batch_item *= 1.2  # 20% overhead for small models

    # Calculate maximum batch size that fits in memory
    max_per_step_batch = int(available_for_activations / memory_per_batch_item)
    max_per_step_batch = max(1, max_per_step_batch)  # At least 1

    # Calculate gradient accumulation needed to reach target effective batch size
    grad_accum = max(
        1, int(math.ceil(target_effective_batch_size / max_per_step_batch))
    )

    # Cap at reasonable limits
    grad_accum = min(grad_accum, 128)  # Maximum accumulation steps

    return grad_accum


def gen_experim(
    hidden_dim,
    gpu_type="V100",
    label=None,
    results_folder=None,
    folder_name=None,
    **overrides,
):
    """
    Generate a scaled transformer experiment configuration.

    Args:
        hidden_dim: Base hidden dimension
        gpu_type: "V100" or "H100" (default: "V100")
        label: Custom label for the experiment (if None, auto-generated)
        results_folder: Custom results folder (if None, uses default from base config)
        **overrides: Any additional config overrides

    Returns:
        List containing a single experiment dictionary in the format expected by the runner
    """
    # Get base config
    base_config = get_base_config()

    # Calculate scaled parameters

    # 1. Scale num_layers proportionally with hidden_dim
    # Use a base ratio: for 32d -> 2 layers, so ratio = 2/32 = 1/16
    base_hidden_dim = 32
    base_num_layers = 2
    layer_scale_ratio = base_num_layers / base_hidden_dim
    num_layers = max(1, int(round(hidden_dim * layer_scale_ratio)))

    # 2. Scale num_heads to keep head dimension close to 32
    target_head_dim = 16
    num_heads = max(1, int(round(hidden_dim / target_head_dim)))
    # Ensure hidden_dim is divisible by num_heads
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    # 3. Calculate total parameters and scale token_limit to 20x parameters
    total_params = calculate_transformer_params(
        hidden_dim,
        num_layers,
        pos_encoding=base_config["pos_encoding"],
        tie_embeddings=base_config["tie_embeddings"],
    )
    token_limit = int(20 * total_params)

    # 4. Estimate gradient accumulation based on GPU memory
    # Use target_effective_batch_size for optimization goals
    target_effective_batch_size = base_config["target_effective_batch_size"]
    grad_accum_steps = estimate_gpu_memory_and_grad_accum(
        hidden_dim,
        num_layers,
        target_effective_batch_size,
        base_config["seq_length"],
        gpu_type,
    )

    # Calculate the actual per-step batch size to use
    per_step_batch_size = max(1, target_effective_batch_size // grad_accum_steps)

    # 5. Handle folder_name parameter (for backward compatibility)
    if folder_name is not None and results_folder is None:
        results_folder = folder_name

    # 6. Generate label if not provided
    if label is None:
        label = f"{hidden_dim}d_generated_{gpu_type.lower()}"

    # Create the experiment configuration
    experiment_config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "token_limit": token_limit,
        "gradient_accumulation_steps": grad_accum_steps,
        "batch_size": per_step_batch_size,  # Override with safe per-step batch size
    }

    # Add results_folder if specified
    if results_folder is not None:
        experiment_config["results_folder"] = results_folder

    # Apply any user overrides
    experiment_config.update(overrides)

    # Create experiment in the expected format
    # Use results_folder as experiment name if provided to avoid double nesting
    if results_folder:
        experiment_name = results_folder
        # Remove results_folder from config since it's now the experiment name
        experiment_config.pop("results_folder", None)
    else:
        experiment_name = f"generated_experiments_{gpu_type.lower()}"

    experiment = {
        "name": experiment_name,
        "subexperiments": [
            {
                "label": label,
                "overrides": experiment_config,
            }
        ],
    }

    return [experiment]


def get_base_config():
    """
    Get the base configuration for experiments.

    Returns:
        Dictionary containing base configuration parameters
    """
    return {
        "dataset": "c4_subset",
        "target_effective_batch_size": 128,  # Target effective batch size for optimization
        "batch_size": 64,  # Default per-step batch size (will be overridden by gen_experim)
        "learning_rate": 0.001 * math.sqrt(4),
        "min_lr": 1e-5,
        "min_lr_multiplier": 0.1,
        "lr_schedule": "cosine_warmup",
        "warmup_frac": 0.02,
        "weight_decay": 0.01,
        "hidden_dim": 64,  # Base hidden dimension
        "num_layers": 4,  # Base number of layers
        "num_heads": 4,
        "dropout": 0.0,
        "seq_length": 128,
        "token_limit": int(5 * 10**7 / 4),  # Convert from old char estimate to tokens
        "pos_encoding": "rotary",
        "init_scheme": "transformer_scaled",
        "stride": 128,
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
        "norm_placement": "pre",
        "results_folder": "new_experiments_folder_1",
        "csv_log_interval": 20,
        "seed": 123,
        # Complete-P (default OFF; non-breaking)
        "enable_completep": False,
        "completep_alpha": 1.0,
        # Base constants for scaling rules
        "n_base": 256,
        "l_base": 2,
        "eta_base": 3.9e-3,
        "wd_base": 0.01,
        "eps_base": 1e-8,
        "use_mup": False,
        "mup_base_width": 128,
        "tie_embeddings": True,  # Default to True for weight tying
        "sgd_momentum": 0.9,  # SGD momentum parameter
        "sgd_nesterov": False,  # SGD Nesterov momentum
    }
