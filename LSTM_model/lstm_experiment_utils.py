# lstm_experiment_utils.py
import copy
import math
import torch


def calculate_lstm_params(
    hidden_size,
    num_layers,
    vocab_size=50257,
    tie_embeddings=True,
):
    """
    Calculate the total number of parameters for an LSTM language model.

    Args:
        hidden_size: Hidden dimension size
        num_layers: Number of LSTM layers
        vocab_size: Vocabulary size (default GPT-2 vocab size)
        tie_embeddings: Whether input and output embeddings are tied (default True)

    Returns:
        Dictionary with total_params and trainable_params
    """
    # Embedding parameters: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size

    # LSTM layer parameters
    # Each LSTM layer has 4 gates (input, forget, cell, output)
    # Each gate has: input_weight (hidden_size * hidden_size) + hidden_weight (hidden_size * hidden_size) + bias (hidden_size)
    # Total per layer: 4 * (hidden_size * hidden_size + hidden_size * hidden_size + hidden_size)
    #                = 4 * (2 * hidden_size^2 + hidden_size)
    #                = 8 * hidden_size^2 + 4 * hidden_size
    params_per_lstm_layer = 8 * hidden_size**2 + 4 * hidden_size
    lstm_params = num_layers * params_per_lstm_layer

    # Final linear layer: hidden_size * vocab_size + vocab_size (bias)
    if tie_embeddings:
        # When tied, we don't count the linear layer weights (they're the same as embedding)
        # But we still have the bias term
        final_layer_params = vocab_size  # Just the bias
    else:
        # Separate weights and bias
        final_layer_params = hidden_size * vocab_size + vocab_size

    total_params = embedding_params + lstm_params + final_layer_params

    # For trainable params calculation with weight tying:
    # When embeddings are tied, we only count the embedding parameters once
    # (the linear layer shares the same weight tensor)
    if tie_embeddings:
        trainable_params = (
            embedding_params + lstm_params + vocab_size
        )  # embedding + lstm + bias
    else:
        trainable_params = total_params  # All parameters are trainable

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "embedding_params": embedding_params,
        "lstm_params": lstm_params,
        "final_layer_params": final_layer_params,
    }


def estimate_lstm_gpu_memory_and_grad_accum(
    hidden_size,
    num_layers,
    batch_size,
    seq_length,
    gpu_type="V100",
    world_size=2,  # Default to 2 GPUs as typical for LSTM training
):
    """
    Estimate GPU memory usage and suggest gradient accumulation steps for LSTM.

    Args:
        hidden_size: Hidden dimension
        num_layers: Number of LSTM layers
        batch_size: Batch size per GPU
        seq_length: Sequence length
        gpu_type: "V100" (16GB) or "H100" (80GB)
        world_size: Number of GPUs (default 2 for LSTM training)

    Returns:
        Suggested gradient accumulation steps
    """
    # GPU memory limits (conservative estimates accounting for CUDA overhead)
    gpu_memory = {
        "V100": 14 * 1024**3,  # ~14GB usable out of 16GB
        "H100": 76 * 1024**3,  # ~76GB usable out of 80GB
    }

    # Calculate LSTM parameters with weight tying
    param_info = calculate_lstm_params(hidden_size, num_layers, tie_embeddings=True)
    params = param_info["trainable_params"]

    # Model memory (4 bytes per param for fp32, but we use mixed precision in practice)
    model_memory = params * 4  # fp32 model weights

    # Gradients and optimizer states (Adam stores gradients + 2 momentum terms)
    # With weight tying, tied parameters share gradients, so memory is correctly estimated
    optimizer_memory = params * 3 * 4  # gradients + 2 adam states

    # LSTM activation memory is different from transformers
    # LSTM needs to store hidden states and cell states for each layer
    # Plus activations for backward pass
    # Rough estimate: batch_size * seq_length * hidden_size * num_layers * 4 bytes * factor
    # Factor accounts for hidden states, cell states, and gradient computation
    lstm_activation_factor = 8  # Higher than transformer due to LSTM state complexity
    activation_memory = (
        batch_size * seq_length * hidden_size * num_layers * 4 * lstm_activation_factor
    )

    # DDP overhead: additional memory for gradient synchronization
    ddp_overhead = model_memory * 0.1 if world_size > 1 else 0

    total_estimated_memory = (
        model_memory + optimizer_memory + activation_memory + ddp_overhead
    )

    available_memory = gpu_memory.get(gpu_type, gpu_memory["V100"])

    # If we exceed 85% of available memory, increase gradient accumulation
    # Use 85% instead of 90% to be more conservative with LSTM's memory patterns
    memory_threshold = 0.85
    if total_estimated_memory > memory_threshold * available_memory:
        # Calculate how much we need to reduce effective batch size
        memory_ratio = total_estimated_memory / (memory_threshold * available_memory)
        suggested_grad_accum = max(1, int(math.ceil(memory_ratio)))
    else:
        suggested_grad_accum = 1

    # Cap gradient accumulation at reasonable limits
    # LSTM training typically needs higher gradient accumulation than transformers
    max_grad_accum = {
        "V100": 64,  # Higher limit for V100 due to memory constraints
        "H100": 16,  # Lower limit for H100 due to better memory
    }
    suggested_grad_accum = min(suggested_grad_accum, max_grad_accum.get(gpu_type, 64))

    return suggested_grad_accum


def gen_lstm_experim(hidden_size, gpu_type="V100", label=None, **overrides):
    """
    Generate a scaled LSTM experiment configuration.

    Args:
        hidden_size: LSTM hidden dimension
        gpu_type: "V100" or "H100" (default: "V100")
        label: Custom label for the experiment (if None, auto-generated)
        **overrides: Any additional config overrides

    Returns:
        List containing a single experiment dictionary in the format expected by the runner
    """
    # Get base config from LSTM experiments
    base_config = get_lstm_base_config()

    # Calculate scaled parameters

    # 1. Scale num_layers proportionally with hidden_size
    # Use a base ratio: for 16d -> 2 layers, so ratio = 2/16 = 1/8
    base_hidden_size = 16
    base_num_layers = 2
    layer_scale_ratio = base_num_layers / base_hidden_size
    num_layers = max(1, int(round(hidden_size * layer_scale_ratio)))

    # 2. Calculate total trainable parameters and scale token limit to 20x parameters
    param_info = calculate_lstm_params(
        hidden_size,
        num_layers,
        tie_embeddings=base_config["tie_embeddings"],
    )
    trainable_params = param_info["trainable_params"]
    # Convert to character limit (using 4:1 ratio for compatibility with existing system)
    max_characters = int(20 * trainable_params * 4)  # 20x params in characters

    # 3. Estimate gradient accumulation based on GPU memory and typical 2-GPU setup
    world_size = 2  # Typical LSTM training setup
    grad_accum_steps = estimate_lstm_gpu_memory_and_grad_accum(
        hidden_size,
        num_layers,
        base_config["batch_size"],
        base_config["sequence_length"],
        gpu_type,
        world_size,
    )

    # 4. Generate label if not provided
    if label is None:
        label = f"lstm_{hidden_size}d_generated_{gpu_type.lower()}"

    # Create the experiment configuration
    experiment_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "max_characters": max_characters,
        "gradient_accumulation_steps": grad_accum_steps,
    }

    # Apply any user overrides
    experiment_config.update(overrides)

    # Create experiment in the expected format
    experiment = {
        "name": f"lstm_generated_experiments_{gpu_type.lower()}",
        "subexperiments": [
            {
                "label": label,
                "overrides": experiment_config,
            }
        ],
    }

    return [experiment]


def get_lstm_base_config():
    """
    Get the base configuration for LSTM experiments.

    Returns:
        Dictionary containing base configuration parameters
    """
    return {
        "data_path": "../Datasets/c4_subset.txt",
        "tokenizer_path": "../gpt2_tokenizer",
        "max_characters": 5 * 1e7,  # Base character limit
        "sequence_length": 128,
        "batch_size": 128,  # Per-GPU batch size
        "hidden_size": 16,  # Base hidden dimension
        "num_layers": 2,  # Base number of layers
        "dropout": 0.0,
        "learning_rate": 0.001 * math.sqrt(4),  # Scale by sqrt of accumulation steps
        "lr_schedule": "cosine_warmup",
        "warmup_frac": 0.01,  # 10% warmup steps
        "min_lr_multiplier": 0.1,  # Min LR as 1% of base LR
        "scheduler_type": "step",
        "step_size": 10,
        "gamma": 0.1,
        "num_epochs": 1,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "lstm-wikitext",
        "wandb_offline": True,
        "print_every": 100,
        "use_gradient_clipping": True,
        "gradient_clip_val": 1.0,
        "results_folder": "Experiments_Folder",
        "csv_log_interval": 20,
        "num_workers": "auto",
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "use_amp": False,
        "gradient_accumulation_steps": 16,  # Base gradient accumulation
        "use_compile": False,
        "seed": 123,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "stride": 128,
        # LSTM-specific dropout settings
        "input_dropout": 0.2,
        "hidden_dropout": 0.1,
        "output_dropout": 0.2,
        "use_layer_norm": True,
        "layer_norm_position": "output",
        "use_mup": True,
        "mup_base_width": 16,
        "tie_embeddings": True,  # Enable weight tying by default
    }


def create_multi_seed_lstm_experiments(base_experiments, seeds):
    """
    Create multiple versions of LSTM experiments with different random seeds.

    Args:
        base_experiments: List of experiment dictionaries
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

                # Add seed to overrides
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


def create_multi_lr_lstm_experiments(base_experiments, learning_rates):
    """
    Create multiple versions of LSTM experiments with different learning rates.

    Args:
        base_experiments: List of experiment dictionaries
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
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["max_characters"] = 129e6
                else:
                    # If neither exists, create overrides
                    new_sub_exp["overrides"] = {
                        "learning_rate": lr,
                        "max_characters": 129e6,
                    }

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    return multi_lr_experiments


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
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["max_characters"] = 129e6
                else:
                    # If neither exists, create overrides with learning rate and max_characters
                    new_sub_exp["overrides"] = {
                        "learning_rate": lr,
                        "max_characters": 129e6,
                    }

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    return multi_lr_experiments


def create_multi_seed_experiments(base_experiments, seeds):
    """
    Create multiple versions of experiments with different random seeds.

    Args:
        base_experiments: List of experiment dictionaries (e.g., LSTM_OPTIMAL_SCALING)
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
