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


# LR sweep experiments use the maximum of a fixed budget (129e6/8) or 5% of full training
def create_multi_lr_experiments(
    base_experiments,
    learning_rates,
    min_tokens=None,
    csv_log_interval_lr_sweep=200,
    min_training_fraction=0.05,
    generate_summary=True,
):
    """
    Create multiple versions of experiments with different learning rates.
    Similar to create_multi_seed_experiments but for learning rates.

    Args:
        base_experiments: List of experiment dictionaries (e.g., LSTM_HIDDEN_DIM_EXPERIMENTS)
        learning_rates: List of learning rate values (e.g., [1e-4, 1e-3, 1e-2])
        min_tokens: Minimum number of tokens to use for max_tokens_training
                   (default: max of 129e6/8 or 5% of full training)
        csv_log_interval_lr_sweep: CSV log interval for LR sweep experiments (default: 200)
        min_training_fraction: Minimum fraction of full training to use (default: 0.05 = 5%)
        generate_summary: If True, create a summary CSV showing best LR for each experiment (default: False)

    Returns:
        List of experiment dictionaries with learning rate variations

    Note:
        If generate_summary=True, after all experiments complete, a summary file will be created
        named "summary_{folder_name}_{timestamp}.csv" containing the best learning rate for each
        base experiment based on lowest final validation loss.
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
            # Get full training tokens from the subexperiment
            full_training_tokens = None
            if "overrides" in sub_exp:
                full_training_tokens = sub_exp["overrides"].get("max_tokens_training")
            elif "config" in sub_exp:
                full_training_tokens = sub_exp["config"].get("max_tokens_training")

            # Calculate the tokens to use for this LR sweep
            if min_tokens is not None:
                tokens_for_lr_sweep = min_tokens
            else:
                # Use maximum of fixed budget (129e6/8) or 5% of full training
                fixed_budget = int(129e6 / 8)
                if full_training_tokens is not None:
                    fraction_tokens = int(full_training_tokens * min_training_fraction)
                    tokens_for_lr_sweep = max(fixed_budget, fraction_tokens)
                else:
                    tokens_for_lr_sweep = fixed_budget

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

                # Add learning rate, max_tokens_training, and csv_log_interval to overrides, and update folder settings
                if "overrides" in new_sub_exp:
                    new_sub_exp["overrides"]["learning_rate"] = lr
                    new_sub_exp["overrides"][
                        "max_tokens_training"
                    ] = tokens_for_lr_sweep
                    new_sub_exp["overrides"][
                        "csv_log_interval"
                    ] = csv_log_interval_lr_sweep
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["overrides"].pop("folder_name", None)
                        new_sub_exp["overrides"].pop("results_folder", None)
                elif "config" in new_sub_exp:
                    new_sub_exp["config"]["learning_rate"] = lr
                    new_sub_exp["config"]["max_tokens_training"] = tokens_for_lr_sweep
                    new_sub_exp["config"][
                        "csv_log_interval"
                    ] = csv_log_interval_lr_sweep
                    # Remove custom folder settings - let the experiment name handle the directory
                    if custom_folder:
                        new_sub_exp["config"].pop("folder_name", None)
                        new_sub_exp["config"].pop("results_folder", None)
                else:
                    # If neither exists, create overrides with learning rate, max_tokens_training, and csv_log_interval
                    overrides_dict = {
                        "learning_rate": lr,
                        "max_tokens_training": tokens_for_lr_sweep,
                        "csv_log_interval": 200,
                    }
                    # Don't add custom folder for new overrides - let experiment name handle it
                    new_sub_exp["overrides"] = overrides_dict

                new_experiment["subexperiments"].append(new_sub_exp)

        multi_lr_experiments.append(new_experiment)

    # Add summary generation metadata if requested
    if generate_summary:
        # Store metadata needed for summary generation
        summary_info = {
            "generate_summary": True,
            "folder_name": custom_folder or "lr_sweep",
            "base_experiments": base_experiments,
            "learning_rates": learning_rates,
        }
        # Add summary info to each experiment for later processing
        for exp in multi_lr_experiments:
            exp["_summary_info"] = summary_info

    return multi_lr_experiments


def generate_lr_sweep_summary(
    experiment_info, results_base_folder="new_experiments_folder_1"
):
    """
    Generate a summary CSV file showing the best learning rate for each base experiment
    based on lowest final validation loss.

    Args:
        experiment_info: Dictionary containing summary metadata from create_multi_lr_experiments
        results_base_folder: Base folder where results are stored

    Returns:
        Path to the created summary file or None if generation failed
    """
    import pandas as pd
    import glob
    import os
    from datetime import datetime

    try:
        folder_name = experiment_info["folder_name"]
        base_experiments = experiment_info["base_experiments"]
        learning_rates = experiment_info["learning_rates"]

        # Create timestamp for summary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"summary_{folder_name}_{timestamp}.csv"

        # Determine the folder where lr sweep results are stored
        lr_sweep_folder = os.path.join(results_base_folder, f"{folder_name}_lr_sweep")
        summary_path = os.path.join(lr_sweep_folder, summary_filename)

        print(f"Generating LR sweep summary for folder: {lr_sweep_folder}")

        # Extract base experiment labels
        base_labels = []
        for exp in base_experiments:
            for sub_exp in exp["subexperiments"]:
                base_labels.append(sub_exp["label"])

        summary_results = []

        # For each base experiment, find the best learning rate
        for base_label in base_labels:
            best_lr = None
            best_val_loss = float("inf")
            lr_results = []

            # Check all learning rates for this base experiment
            for lr in learning_rates:
                # Format learning rate for filename matching (EXACT same logic as create_multi_lr_experiments)
                if lr >= 1:
                    lr_str = f"{lr:.0f}"
                else:
                    import math

                    # Use clear scientific notation: 10e-1, 10e-2, 10e-3, etc.
                    log_lr = math.log10(lr)

                    # Check if it's close to a nice power of 10
                    if (
                        abs(log_lr - round(log_lr)) < 0.01
                    ):  # Very close to integer power
                        exponent = int(round(log_lr))
                        lr_str = f"10e{abs(exponent)}"  # Use positive exponent format: 10e3 for 10^-3
                    else:
                        # For non-integer powers, use coefficient notation
                        exponent = math.floor(log_lr)
                        coefficient = lr / (10**exponent)
                        if abs(coefficient - round(coefficient)) < 0.01:
                            lr_str = f"{round(coefficient):.0f}e{abs(exponent)}"
                        else:
                            # Handle decimal coefficients: multiply by 10 and remove decimal point
                            # e.g., 1.8 becomes 18, 3.2 becomes 32, 5.6 becomes 56
                            coef_scaled = round(coefficient * 10)
                            lr_str = f"{coef_scaled}e{abs(exponent)}"

                # Construct expected CSV filename
                csv_filename = f"{base_label}_lr_{lr_str}.csv"
                csv_path = os.path.join(lr_sweep_folder, csv_filename)

                # Read the CSV and get final validation loss
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if "validation_loss" in df.columns and len(df) > 0:
                            # Get the final (last) validation loss that's not NaN
                            valid_losses = df["validation_loss"].dropna()
                            if len(valid_losses) > 0:
                                final_val_loss = valid_losses.iloc[-1]
                                lr_results.append((lr, final_val_loss))

                                # Track best learning rate
                                if final_val_loss < best_val_loss:
                                    best_val_loss = final_val_loss
                                    best_lr = lr
                            else:
                                print(
                                    f"Warning: No valid validation losses found in {csv_filename}"
                                )
                        else:
                            print(
                                f"Warning: No validation_loss column found in {csv_filename}"
                            )
                    except Exception as e:
                        print(f"Warning: Could not read {csv_filename}: {e}")
                else:
                    print(f"Warning: Expected file not found: {csv_path}")

            # Add results for this base experiment
            if best_lr is not None:
                summary_results.append(
                    {
                        "experiment": base_label,
                        "best_learning_rate": best_lr,
                        "best_validation_loss": best_val_loss,
                        "num_lr_tested": len(lr_results),
                    }
                )
                print(
                    f"{base_label}: best_lr={best_lr:.2e}, best_val_loss={best_val_loss:.4f}"
                )
            else:
                print(f"Warning: No valid results found for {base_label}")
                summary_results.append(
                    {
                        "experiment": base_label,
                        "best_learning_rate": "N/A",
                        "best_validation_loss": "N/A",
                        "num_lr_tested": 0,
                    }
                )

        # Create summary DataFrame and save
        if summary_results:
            summary_df = pd.DataFrame(summary_results)

            # Ensure output directory exists
            os.makedirs(lr_sweep_folder, exist_ok=True)

            # Save summary CSV
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Summary saved to: {summary_path}")

            # Also print summary to console
            print("\n" + "=" * 60)
            print("LEARNING RATE SWEEP SUMMARY")
            print("=" * 60)
            for _, row in summary_df.iterrows():
                if row["best_learning_rate"] != "N/A":
                    print(f"{row['experiment']:30} {row['best_learning_rate']:12.2e}")
                else:
                    print(f"{row['experiment']:30} {'N/A':>12}")
            print("=" * 60)

            return summary_path
        else:
            print("Warning: No valid results found for summary generation")
            return None

    except Exception as e:
        print(f"Error generating LR sweep summary: {e}")
        import traceback

        traceback.print_exc()
        return None


def calculate_transformer_params(
    hidden_dim,
    num_layers,
    vocab_size=50257,
    seq_length=128,
    pos_encoding="rotary",
    tie_embeddings=True,
    ff_ratio=4,
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
    # - Feed-forward: 2 * hidden_dim * (ff_ratio * hidden_dim) = 2 * ff_ratio * hidden_dim^2
    # - Layer norms: 2 * hidden_dim (small, can be ignored for scaling)
    params_per_layer = 4 * hidden_dim**2 + 2 * ff_ratio * hidden_dim**2
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
    ff_ratio=4,
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
    # - Feed-forward: 2 * hidden_dim * (ff_ratio * hidden_dim) = 2 * ff_ratio * hidden_dim^2
    # - Layer norms: 2 * hidden_dim (small, can be ignored for scaling)
    params_per_layer = 4 * hidden_dim**2 + 2 * ff_ratio * hidden_dim**2
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
    hidden_dim,
    num_layers,
    target_effective_batch_size,
    seq_length,
    gpu_type="V100",
    ff_ratio=4,
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
    target_head_dim = 16  # Match gen_experim for consistency
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
    ff_per_batch = seq_length * (ff_ratio * hidden_dim) * num_layers * 4

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
    scaling_law="chinchilla",
    **overrides,
):
    """
    Generate a scaled transformer experiment configuration.

    Args:
        hidden_dim: Base hidden dimension
        gpu_type: "V100" or "H100" (default: "V100")
        label: Custom label for the experiment (if None, auto-generated)
        results_folder: Custom results folder (if None, uses default from base config)
        scaling_law: Token scaling approach:
            - "chinchilla" (default): D ≈ 20*N (tokens scale linearly with params)
            - "kaplan": D ≈ N^0.74 (tokens scale sublinearly with params)
        **overrides: Any additional config overrides

    Returns:
        List containing a single experiment dictionary in the format expected by the runner
    """
    # Get base config
    base_config = get_base_config()

    # Get pos_encoding early (needed for num_heads calculation)
    pos_encoding = overrides.get("pos_encoding", base_config["pos_encoding"])

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
    # AND that head_dim is even (required for rotary embeddings)
    while hidden_dim % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    # If using rotary encoding, ensure head_dim is even
    if pos_encoding == "rotary":
        while (hidden_dim // num_heads) % 2 != 0 and num_heads > 1:
            num_heads -= 1
            # Re-check divisibility after adjustment
            while hidden_dim % num_heads != 0 and num_heads > 1:
                num_heads -= 1

    # 3. Calculate total parameters and scale max_tokens based on scaling law
    # Use user overrides for tie_embeddings and ff_ratio if provided
    tie_embeddings = overrides.get("tie_embeddings", base_config["tie_embeddings"])
    ff_ratio = overrides.get("ff_ratio", base_config["ff_ratio"])

    total_params = calculate_transformer_params(
        hidden_dim,
        num_layers,
        pos_encoding=pos_encoding,
        tie_embeddings=tie_embeddings,
        ff_ratio=ff_ratio,
    )

    # Determine max_tokens_training based on scaling law
    # Allow explicit max_tokens_training or token_to_param_ratio override to take precedence
    if "max_tokens_training" in overrides:
        max_tokens_training = overrides["max_tokens_training"]
    elif "token_to_param_ratio" in overrides:
        token_to_param_ratio = overrides["token_to_param_ratio"]
        max_tokens_training = int(token_to_param_ratio * total_params)
    elif scaling_law == "kaplan":
        # Kaplan et al. (2020): D_opt ≈ N^0.74
        # where D is tokens and N is parameters
        # This means larger models need proportionally FEWER tokens per parameter
        max_tokens_training = int(831.4 * total_params**0.74)
    else:  # "chinchilla" or default
        # Chinchilla (Hoffmann et al., 2022): D_opt ≈ N (approximately 20:1 ratio)
        # Tokens scale linearly with parameters
        token_to_param_ratio = base_config["token_to_param_ratio"]
        max_tokens_training = int(token_to_param_ratio * total_params)

    # 4. Estimate gradient accumulation based on GPU memory
    # Use target_effective_batch_size for optimization goals
    # Check if user provided target_effective_batch_size and seq_length in overrides first
    target_effective_batch_size = overrides.get(
        "target_effective_batch_size", base_config["target_effective_batch_size"]
    )
    seq_length = overrides.get("seq_length", base_config["seq_length"])

    grad_accum_steps = estimate_gpu_memory_and_grad_accum(
        hidden_dim,
        num_layers,
        target_effective_batch_size,
        seq_length,
        gpu_type,
        ff_ratio,
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
        "max_tokens_training": max_tokens_training,
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
        Dictionary containing base configuration parameters including:

        Dataset Configuration:
        - data_path: Path to the dataset file
        - max_tokens_training: Maximum number of tokens for training (validation tokens added on top)
        - train_split: Fraction of dataset for training (default: 0.9 = 90%)
        - val_split: Fraction of dataset for validation (default: 0.1 = 10%)
        - fixed_val_tokens: Fixed number of tokens for validation set (default: None)
        - use_streaming_dataset: Use streaming dataset for large files (default: False)

        Model Configuration:
        - hidden_dim, num_layers, num_heads: Model architecture parameters
        - seq_length, pos_encoding: Sequence and positional encoding settings
        - activation, norm_type, norm_placement: Activation and normalization settings

        Training Configuration:
        - learning_rate, lr_schedule, warmup_frac: Learning rate settings
        - batch_size, target_effective_batch_size: Batch size configuration
        - gradient_accumulation_steps: Gradient accumulation for large effective batch sizes
        - optimizer, weight_decay: Optimizer configuration

        Other Configuration:
        - seed: Random seed for reproducibility
        - max_epochs, min_epochs: Training duration
        - use_gradient_clipping, gradient_clip_val: Gradient clipping settings
        - Complete-P and muP scaling parameters
    """
    return {
        "data_path": "Datasets/c4_subset_large.txt",  # Actual dataset file path
        "max_tokens_training": int(
            5 * 1e7 / 4
        ),  # Maximum number of tokens for training (validation tokens added on top)
        "target_effective_batch_size": 64,  # Target effective batch size for optimization
        "batch_size": 64,  # Default per-step batch size (will be overridden by gen_experim)
        "learning_rate": 0.001 * math.sqrt(4),
        "min_lr": 1e-5,
        "min_lr_multiplier": 0.1,
        "lr_schedule": "cosine_warmup",
        "warmup_frac": 0.1,
        "weight_decay": 0.01,
        "hidden_dim": 64,  # Base hidden dimension
        "num_layers": 4,  # Base number of layers
        "num_heads": 4,
        "dropout": 0.0,
        "seq_length": 128,
        "pos_encoding": "rotary",
        "init_scheme": "bert_gpt",
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
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "activation": "swiglu",
        "norm_type": "rms",
        "norm_placement": "pre",
        "results_folder": "new_experiments_folder_1",
        "csv_log_interval": 200,
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
        "use_amp": False,
        # Dataset split configuration
        "train_split": 0.9,  # Fraction of dataset to use for training (default: 90%)
        "val_split": 0.1,  # Fraction of dataset to use for validation (default: 10%)
        "fixed_val_tokens": int(
            500e3
        ),  # Fixed number of tokens for validation set (None = use percentage split)
        "char_to_token_ratio": 4.0,  # Character-to-token ratio for dataset loading (e.g., 4.0 = load 4 chars per expected token)
        "use_streaming_dataset": True,  # Use streaming dataset for large files (default: False = memory-based loading)
        "ff_ratio": 2.5,  # Feedforward dimension to model dimension ratio (default: 4)
        "modern_bias_0": True,  # Modern architecture: remove biases from layers followed by normalization (default: False)
        "token_to_param_ratio": 20,  # Chinchilla: D ≈ 20*N (linear scaling). Kaplan: D ≈ N^0.74 (sublinear)
    }
