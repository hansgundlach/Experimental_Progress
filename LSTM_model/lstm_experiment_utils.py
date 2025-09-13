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
    effective_batch_size,
    seq_length,
    gpu_type="V100",
    world_size=2,  # Default to 2 GPUs as typical for LSTM training
):
    """
    Estimate GPU memory usage and suggest gradient accumulation steps for LSTM.

    FIXED: Now maintains constant effective batch size by adjusting per-step batch size.

    Args:
        hidden_size: Hidden dimension
        num_layers: Number of LSTM layers
        effective_batch_size: Target effective batch size (constant across experiments)
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
    vocab_size = 50257  # GPT-2 vocab size

    # Model memory (4 bytes per param for fp32, but we use mixed precision in practice)
    model_memory = params * 4  # fp32 model weights

    # Gradients and optimizer states (Adam stores gradients + 2 momentum terms)
    # With weight tying, tied parameters share gradients, so memory is correctly estimated
    optimizer_memory = params * 3 * 4  # gradients + 2 adam states

    available_memory = gpu_memory.get(gpu_type, gpu_memory["V100"])

    # Use 70% of available memory as threshold (very conservative due to memory spikes)
    memory_threshold = 0.70 * available_memory

    # FIXED APPROACH: Find the gradient accumulation needed to fit in memory
    # Start with grad_accum = 1 and increase until memory fits
    for grad_accum in range(1, 65):  # Cap at 64
        # Calculate per-step batch size (what actually goes through the model)
        per_step_batch_size = effective_batch_size // grad_accum

        if per_step_batch_size < 1:
            # If effective batch size becomes too small, use the previous grad_accum
            grad_accum = grad_accum - 1
            break

        # CRITICAL: Final layer (logits) tensor for LSTM
        # Output tensor: [per_step_batch_size, seq_length, vocab_size]
        # This is often the largest single tensor in LSTM models too
        final_layer_output = (
            per_step_batch_size * seq_length * vocab_size * 4
        )  # 4 bytes per float32
        final_layer_gradients = final_layer_output  # Gradients have same size
        final_layer_memory = final_layer_output + final_layer_gradients

        # LSTM activation memory is different from transformers
        # LSTM needs to store hidden states and cell states for each layer
        # Plus activations for backward pass
        lstm_hidden_states = (
            per_step_batch_size * seq_length * hidden_size * num_layers * 4
        )  # hidden states
        lstm_cell_states = (
            per_step_batch_size * seq_length * hidden_size * num_layers * 4
        )  # cell states
        lstm_gate_activations = (
            per_step_batch_size * seq_length * hidden_size * num_layers * 4 * 4
        )  # 4 gates

        # Gradient storage for LSTM activations
        lstm_gradients = lstm_hidden_states + lstm_cell_states + lstm_gate_activations

        total_lstm_activation_memory = (
            lstm_hidden_states
            + lstm_cell_states
            + lstm_gate_activations
            + lstm_gradients
        )

        # DDP overhead: additional memory for gradient synchronization
        ddp_overhead = model_memory * 0.1 if world_size > 1 else 0

        # 5% overhead for misc tensors and CUDA overhead
        overhead = 0.05 * (
            model_memory
            + optimizer_memory
            + total_lstm_activation_memory
            + final_layer_memory
            + ddp_overhead
        )

        total_estimated_memory = (
            model_memory
            + optimizer_memory
            + total_lstm_activation_memory
            + final_layer_memory
            + ddp_overhead
            + overhead
        )

        # Check if this gradient accumulation level fits in memory
        if total_estimated_memory <= memory_threshold:
            return grad_accum

    # If we couldn't fit even with max gradient accumulation, return max
    return 64


def gen_lstm_experim(
    hidden_size,
    gpu_type="V100",
    label=None,
    results_folder=None,
    folder_name=None,
    **overrides,
):
    """
    Generate a scaled LSTM experiment configuration.

    Args:
        hidden_size: LSTM hidden dimension
        gpu_type: "V100" or "H100" (default: "V100")
        label: Custom label for the experiment (if None, auto-generated)
        results_folder: Custom results folder (if None, uses default from base config)
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
    # Use target_effective_batch_size parameter (constant across experiments)
    world_size = 2  # Typical LSTM training setup
    target_effective_batch_size = base_config["target_effective_batch_size"]
    grad_accum_steps = estimate_lstm_gpu_memory_and_grad_accum(
        hidden_size,
        num_layers,
        target_effective_batch_size,
        base_config["sequence_length"],
        gpu_type,
        world_size,
    )

    # Calculate the actual per-step batch size to use
    per_step_batch_size = max(1, target_effective_batch_size // grad_accum_steps)

    # 4. Handle folder_name parameter (for backward compatibility)
    if folder_name is not None and results_folder is None:
        results_folder = folder_name

    # 5. Generate label if not provided
    if label is None:
        label = f"lstm_{hidden_size}d_generated_{gpu_type.lower()}"

    # Create the experiment configuration
    experiment_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "max_characters": max_characters,
        "gradient_accumulation_steps": grad_accum_steps,
        "batch_size": per_step_batch_size,  # Override with safe per-step batch size
    }

    # Add results_folder if specified
    if results_folder is not None:
        experiment_config["results_folder"] = results_folder

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
        "target_effective_batch_size": 512,  # Target effective batch size for optimization
        "batch_size": 32,  # Default per-step batch size (will be overridden by gen_lstm_experim)
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
        "results_folder": "../new_experiments_folder",
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
                # Format learning rate for filename-safe label with clear scientific notation
                import math
                if lr >= 1:
                    lr_str = f"{lr:.0f}"
                else:
                    # Use clear scientific notation: 10e-1, 10e-2, 10e-3, etc.
                    log_lr = math.log10(lr)
                    
                    # Check if it's close to a nice power of 10
                    if abs(log_lr - round(log_lr)) < 0.01:  # Very close to integer power
                        exponent = int(round(log_lr))
                        lr_str = f"10e{exponent:+d}"  # +d ensures +/- sign
                    else:
                        # For non-integer powers, use coefficient notation
                        exponent = math.floor(log_lr)
                        coefficient = lr / (10 ** exponent)
                        if abs(coefficient - round(coefficient)) < 0.01:
                            lr_str = f"{round(coefficient):.0f}e{exponent:+d}"
                        else:
                            lr_str = f"{coefficient:.1f}e{exponent:+d}"

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
                    if abs(log_lr - round(log_lr)) < 0.01:  # Very close to integer power
                        exponent = int(round(log_lr))
                        lr_str = f"10e{exponent:+d}"  # +d ensures +/- sign
                    else:
                        # For non-integer powers, use coefficient notation
                        exponent = math.floor(log_lr)
                        coefficient = lr / (10 ** exponent)
                        if abs(coefficient - round(coefficient)) < 0.01:
                            lr_str = f"{round(coefficient):.0f}e{exponent:+d}"
                        else:
                            lr_str = f"{coefficient:.1f}e{exponent:+d}"

                new_sub_exp["label"] = f"{original_label}_lr_{lr_str}"

                # Add learning rate and max_characters to overrides, and remove folder settings
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["learning_rate"] = lr
                    new_sub_exp["overrides"]["max_characters"] = 129e6
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["overrides"].pop("folder_name", None)
                        new_sub_exp["overrides"].pop("results_folder", None)
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["max_characters"] = 129e6
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["config"].pop("folder_name", None)
                        new_sub_exp["config"].pop("results_folder", None)
                else:
                    # If neither exists, create overrides with learning rate and max_characters
                    # Don't add custom folder for new overrides - let experiment name handle it
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
