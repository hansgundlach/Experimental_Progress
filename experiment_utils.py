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
        # Create a new experiment group for each base experiment
        new_experiment = {
            "name": f"{experiment['name']}_lr_sweep",
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
                # Format learning rate for filename-safe label
                if lr >= 1:
                    lr_str = f"{lr:.0f}"
                elif lr >= 0.01:
                    lr_str = f"{lr:.3f}".rstrip("0").rstrip(".")
                else:
                    # For very small learning rates, use scientific notation
                    lr_str = f"{lr:.1e}".replace("-", "m").replace("+", "p")

                new_sub_exp["label"] = f"{original_label}_lr_{lr_str}"

                # Add learning rate and max_characters to overrides
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["learning_rate"] = lr
                    new_sub_exp["overrides"]["max_characters"] = 129e6
                    new_sub_exp["overrides"][
                        "wikitext_limit"
                    ] = 129e6  # Same as max_characters
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["max_characters"] = 129e6
                    new_sub_exp["config"][
                        "wikitext_limit"
                    ] = 129e6  # Same as max_characters
                else:
                    # If neither exists, create overrides with learning rate and max_characters
                    new_sub_exp["overrides"] = {
                        "learning_rate": lr,
                        "max_characters": 129e6,
                        "wikitext_limit": 129e6,  # Same as max_characters
                    }

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    return multi_lr_experiments


def calculate_transformer_params(
    hidden_dim, num_layers, vocab_size=50257, seq_length=128, pos_encoding="rotary"
):
    """
    Calculate the total number of parameters for a transformer model.

    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        vocab_size: Vocabulary size (default GPT-2 vocab size)
        seq_length: Sequence length (for learned positional embeddings)
        pos_encoding: Type of positional encoding ("rotary", "learned", "sinusoidal")

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

    # Final linear layer: hidden_dim * vocab_size
    final_layer_params = hidden_dim * vocab_size

    total_params = embedding_params + pos_emb_params + layer_params + final_layer_params
    return total_params


def estimate_gpu_memory_and_grad_accum(
    hidden_dim, num_layers, batch_size, seq_length, gpu_type="V100"
):
    """
    Estimate GPU memory usage and suggest gradient accumulation steps.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        batch_size: Batch size
        seq_length: Sequence length
        gpu_type: "V100" (16GB) or "H100" (80GB)

    Returns:
        Suggested gradient accumulation steps
    """
    # GPU memory limits (conservative estimates accounting for CUDA overhead)
    gpu_memory = {
        "V100": 14 * 1024**3,  # ~14GB usable out of 16GB
        "H100": 76 * 1024**3,  # ~76GB usable out of 80GB
    }

    # Rough memory estimation (very approximate)
    # Model parameters (4 bytes per param for fp32, but we use mixed precision)
    params = calculate_transformer_params(hidden_dim, num_layers)
    model_memory = params * 4  # fp32 model weights

    # Gradients and optimizer states (Adam stores gradients + 2 momentum terms)
    optimizer_memory = params * 3 * 4  # gradients + 2 adam states

    # Activation memory (depends on batch size and sequence length)
    # Rough estimate: batch_size * seq_length * hidden_dim * num_layers * 4 bytes * some_factor
    activation_memory = (
        batch_size * seq_length * hidden_dim * num_layers * 4 * 6
    )  # 6x factor for intermediate activations

    total_estimated_memory = model_memory + optimizer_memory + activation_memory

    available_memory = gpu_memory.get(gpu_type, gpu_memory["V100"])

    # If we exceed 90% of available memory, increase gradient accumulation
    if total_estimated_memory > 0.9 * available_memory:
        # Calculate how much we need to reduce batch size
        memory_ratio = total_estimated_memory / (0.9 * available_memory)
        suggested_grad_accum = max(1, int(math.ceil(memory_ratio)))
    else:
        suggested_grad_accum = 1

    # Cap gradient accumulation at reasonable limits
    max_grad_accum = (
        32 if gpu_type == "V100" else 8
    )  # H100 has more memory, less need for high grad accum
    suggested_grad_accum = min(suggested_grad_accum, max_grad_accum)

    return suggested_grad_accum


def gen_experim(hidden_dim, gpu_type="V100", label=None, **overrides):
    """
    Generate a scaled transformer experiment configuration.

    Args:
        hidden_dim: Base hidden dimension
        gpu_type: "V100" or "H100" (default: "V100")
        label: Custom label for the experiment (if None, auto-generated)
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
    target_head_dim = 32
    num_heads = max(1, int(round(hidden_dim / target_head_dim)))
    # Ensure hidden_dim is divisible by num_heads
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    # 3. Calculate total parameters and scale wikitext_limit to 20x parameters
    total_params = calculate_transformer_params(
        hidden_dim, num_layers, pos_encoding=base_config["pos_encoding"]
    )
    wikitext_limit = int(20 * total_params)

    # 4. Estimate gradient accumulation based on GPU memory
    grad_accum_steps = estimate_gpu_memory_and_grad_accum(
        hidden_dim,
        num_layers,
        base_config["batch_size"],
        base_config["seq_length"],
        gpu_type,
    )

    # 5. Generate label if not provided
    if label is None:
        label = f"{hidden_dim}d_generated_{gpu_type.lower()}"

    # Create the experiment configuration
    experiment_config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "wikitext_limit": wikitext_limit,
        "gradient_accumulation_steps": grad_accum_steps,
    }

    # Apply any user overrides
    experiment_config.update(overrides)

    # Create experiment in the expected format
    experiment = {
        "name": f"generated_experiments_{gpu_type.lower()}",
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
        "batch_size": 32,  # physical batch size 256
        "learning_rate": 0.001 * math.sqrt(4),
        "min_lr": 1e-5,
        "min_lr_multiplier": 0.1,
        "lr_schedule": "cosine_warmup",
        "warmup_frac": 0.01,
        "weight_decay": 0.01,
        "hidden_dim": 64,  # Base hidden dimension
        "num_layers": 4,  # Base number of layers
        "num_heads": 4,
        "dropout": 0.0,
        "seq_length": 128,
        "wikitext_limit": 5 * 10**7,
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
        "results_folder": "Former_Experiments_Folder",
        "csv_log_interval": 20,
        "seed": 789,
        # Complete-P (default OFF; non-breaking)
        "enable_completep": False,
        "completep_alpha": 1.0,
        # Base constants for scaling rules
        "n_base": 256,
        "l_base": 2,
        "eta_base": 3.9e-3,
        "wd_base": 0.10,
        "eps_base": 1e-16,
    }