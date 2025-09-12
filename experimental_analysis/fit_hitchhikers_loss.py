


import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass
class RunSpec:
    csv_path: str
    num_parameters: int


@dataclass
class FitResult:
    E: float  # Log-space parameter (actual irreducible loss is exp(E))
    A: float  # Log-space parameter (actual coefficient is exp(A))
    alpha: float  # Scaling exponent for parameters (should be negative)
    B: float  # Log-space parameter (actual coefficient is exp(B))
    beta: float  # Scaling exponent for data (should be negative)
    huber_beta: float
    num_points: int
    final_loss: float
    # Derived quantities for easier interpretation
    exp_E: float  # exp(E) - irreducible loss
    exp_A: float  # exp(A) - parameter scaling coefficient
    exp_B: float  # exp(B) - data scaling coefficient


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def read_and_aggregate_csv(
    csv_path: str,
    tokens_per_step: int,
    loss_column: str,
    use_tokens_column: bool = True,
    ignore_first_percent: float = 0.0,
) -> pd.DataFrame:
    """
    Reads a CSV with columns including 'step' and the given loss column, averages over duplicate steps,
    filters out step == 0 (since D=0 tokens is not valid for the model), and returns a DataFrame with
    columns: step, tokens (D), loss.

    Args:
        csv_path: Path to the CSV file
        tokens_per_step: Tokens per step (used only if use_tokens_column=False)
        loss_column: Name of the loss column
        use_tokens_column: If True, read tokens directly from 'tokens' column in CSV.
                          If False, calculate tokens as step * tokens_per_step.
    """
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'step' column.")
    if loss_column not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a '{loss_column}' column.")

    if use_tokens_column:
        if "tokens" not in df.columns:
            raise ValueError(
                f"CSV {csv_path} must contain a 'tokens' column when use_tokens_column=True."
            )

        # Group by step to average repeated evaluations/log lines, including tokens
        grouped = (
            df[["step", "tokens", loss_column]]
            .groupby("step", as_index=False)
            .mean()
            .sort_values("step")
        )

        # Filter out entries where tokens <= 0 to avoid division by zero in D**beta
        grouped = grouped[grouped["tokens"] > 0].copy()
    else:
        # Original behavior: calculate tokens from steps
        grouped = (
            df[["step", loss_column]]
            .groupby("step", as_index=False)
            .mean()
            .sort_values("step")
        )

        # Filter out step==0 to avoid division by zero in D**beta
        grouped = grouped[grouped["step"] > 0].copy()
        grouped["tokens"] = grouped["step"].astype(np.float64) * float(tokens_per_step)

    grouped.rename(columns={loss_column: "loss"}, inplace=True)

    # Ignore first X percent of data if specified
    if ignore_first_percent > 0.0:
        total_rows = len(grouped)
        rows_to_ignore = int(total_rows * ignore_first_percent / 100.0)
        if rows_to_ignore > 0:
            grouped = grouped.iloc[rows_to_ignore:].copy()
            print(
                f"Ignored first {rows_to_ignore}/{total_rows} rows ({ignore_first_percent:.1f}%) from {csv_path}"
            )

    return grouped[["step", "tokens", "loss"]]


def build_dataset(
    pairs: Sequence[RunSpec],
    tokens_per_step: int,
    loss_column: str,
    use_tokens_column: bool = True,
    ignore_first_percent: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (csv_path, N), load and aggregate, then create arrays of N, D (tokens), and observed loss.

    Args:
        pairs: Sequence of RunSpec objects with csv_path and num_parameters
        tokens_per_step: Tokens per step (used only if use_tokens_column=False)
        loss_column: Name of the loss column
        use_tokens_column: If True, read tokens directly from 'tokens' column in CSV.
                          If False, calculate tokens as step * tokens_per_step.
        ignore_first_percent: Percentage of data to ignore from the beginning (0.0-100.0).

    Returns:
      - N_all: shape [M]
      - D_all: shape [M]
      - y_all: shape [M]
    """
    N_list: List[float] = []
    D_list: List[float] = []
    y_list: List[float] = []

    for spec in pairs:
        agg = read_and_aggregate_csv(
            spec.csv_path,
            tokens_per_step,
            loss_column,
            use_tokens_column,
            ignore_first_percent,
        )
        N_vals = np.full(
            shape=(len(agg),), fill_value=float(spec.num_parameters), dtype=np.float64
        )
        D_vals = agg["tokens"].to_numpy(dtype=np.float64)
        y_vals = agg["loss"].to_numpy(dtype=np.float64)

        # Filter any non-finite entries just in case
        mask = np.isfinite(D_vals) & np.isfinite(y_vals)
        N_list.append(N_vals[mask])
        D_list.append(D_vals[mask])
        y_list.append(y_vals[mask])

    if not N_list:
        raise ValueError("No data points built from the provided pairs.")

    N_all = np.concatenate(N_list, axis=0)
    D_all = np.concatenate(D_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return N_all, D_all, y_all


# --------------------------------------------------------------------------------------
# Model and fitting (scipy curve_fit)
# --------------------------------------------------------------------------------------


def validation_loss_model(
    N: np.ndarray,
    D: np.ndarray,
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
) -> np.ndarray:
    """
    L(N, D) = exp(E) + exp(A) * N^(-alpha) + exp(B) * D^(-beta)
    Following Hoffmann et al. (2022) form: L(f) = e^E + e^A * #params(f)^α + e^B * #toks(f)^β
    where alpha and beta are negative (scaling exponents), E, A, B are log-space parameters.

    Args:
        N: Number of parameters (shape: [M])
        D: Number of tokens (shape: [M])
        E: Log-space parameter for irreducible loss
        A: Log-space parameter for parameter scaling coefficient
        alpha: Scaling exponent for parameters (should be negative)
        B: Log-space parameter for data scaling coefficient
        beta: Scaling exponent for data (should be negative)

    Returns:
        Predicted loss values (shape: [M])
    """
    # Guard against numerical issues with extremely small N or D
    N_clamped = np.maximum(N, 1.0)
    D_clamped = np.maximum(D, 1.0)

    return (
        np.exp(E)
        + np.exp(A) * np.power(N_clamped, alpha)
        + np.exp(B) * np.power(D_clamped, beta)
    )


def fit_parameters(
    N_all: np.ndarray,
    D_all: np.ndarray,
    y_all: np.ndarray,
    huber_beta: float = 0.1,
    adam_lr: float = 0.05,
    adam_steps: int = 4000,
    lbfgs_steps: int = 200,
    seed: int = 42,
) -> FitResult:
    """
    Fit the validation loss model using scipy's curve_fit.

    Args:
        N_all: Number of parameters for each data point
        D_all: Number of tokens for each data point
        y_all: Observed loss values
        huber_beta: Not used in curve_fit (kept for compatibility)
        adam_lr: Not used in curve_fit (kept for compatibility)
        adam_steps: Not used in curve_fit (kept for compatibility)
        lbfgs_steps: Not used in curve_fit (kept for compatibility)
        seed: Random seed for reproducibility

    Returns:
        FitResult with fitted parameters and diagnostics
    """
    np.random.seed(seed)

    # Initial guesses for exponential form: L = exp(E) + exp(A) * N^alpha + exp(B) * D^beta
    # E_init: log of baseline loss level
    E_init = float(np.log(max(np.percentile(y_all, 10), 1e-6)))
    # A_init and B_init: log of scaling coefficients
    range_y = max(float(np.percentile(y_all, 90) - np.percentile(y_all, 10)), 1e-3)
    A_init = float(np.log(range_y))
    B_init = float(np.log(range_y))
    # alpha and beta: scaling exponents (should be negative for compute scaling)
    alpha_init = -0.5  # Negative for parameter scaling
    beta_init = -0.2  # Negative for data scaling

    initial_guess = [E_init, A_init, alpha_init, B_init, beta_init]

    # Define the function for curve_fit (needs to take parameters as separate arguments)
    def model_func(data, E, A, alpha, B, beta):
        N, D = data
        return validation_loss_model(N, D, E, A, alpha, B, beta)

    # Prepare data for curve_fit (combine N and D into a single array)
    data = np.vstack([N_all, D_all])

    try:
        # Use curve_fit with bounds to help with convergence
        # Set reasonable bounds for the parameters
        bounds = (
            [-10, -10, -2, -10, -2],  # Lower bounds
            [10, 10, 0, 10, 0],  # Upper bounds
        )

        popt, pcov = curve_fit(
            model_func,
            data,
            y_all,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000,  # Maximum number of function evaluations
            method="trf",  # Trust Region Reflective algorithm
        )

        E_val, A_val, alpha_val, B_val, beta_val = popt

        # Calculate final loss (MSE)
        y_pred = validation_loss_model(
            N_all, D_all, E_val, A_val, alpha_val, B_val, beta_val
        )
        final_loss = float(np.mean((y_all - y_pred) ** 2))

        result = FitResult(
            E=float(E_val),
            A=float(A_val),
            alpha=float(alpha_val),
            B=float(B_val),
            beta=float(beta_val),
            huber_beta=float(huber_beta),
            num_points=int(y_all.shape[0]),
            final_loss=final_loss,
            exp_E=float(np.exp(E_val)),
            exp_A=float(np.exp(A_val)),
            exp_B=float(np.exp(B_val)),
        )

    except Exception as e:
        print(f"Warning: curve_fit failed with error: {e}")
        print("Falling back to initial guess values")

        # Fallback to initial guess
        result = FitResult(
            E=float(E_init),
            A=float(A_init),
            alpha=float(alpha_init),
            B=float(B_init),
            beta=float(beta_init),
            huber_beta=float(huber_beta),
            num_points=int(y_all.shape[0]),
            final_loss=float("inf"),
            exp_E=float(np.exp(E_init)),
            exp_A=float(np.exp(A_init)),
            exp_B=float(np.exp(B_init)),
        )

    return result


# --------------------------------------------------------------------------------------
# Public API (programmatic use; no CLI)
# --------------------------------------------------------------------------------------


def fit_validation_loss_from_pairs(
    files_and_N: Sequence[Tuple[str, int]],
    *,
    tokens_per_step: int = 0,
    loss_column: str = "validation_loss",
    use_tokens_column: bool = True,
    ignore_first_percent: float = 0.0,
    huber_beta: float = 0.1,
    adam_lr: float = 0.05,
    adam_steps: int = 4000,
    lbfgs_steps: int = 200,
    seed: int = 42,
) -> FitResult:
    """
    Programmatic entry point to fit the validation loss model across multiple runs.

    Args:
        files_and_N: Sequence of (csv_path, num_parameters) pairs.
        tokens_per_step: Tokens per training step (used only if use_tokens_column=False, default: 0).
        loss_column: Column name to use as target loss (default: 'validation_loss').
        use_tokens_column: If True, read tokens directly from 'tokens' column in CSV.
                          If False, calculate tokens as step * tokens_per_step (default: True).
        ignore_first_percent: Percentage of data to ignore from the beginning (0.0-100.0).
                             Useful for ignoring early training steps that don't follow scaling laws.
        huber_beta: Huber loss beta threshold.
        adam_lr: Adam learning rate.
        adam_steps: Number of Adam steps.
        lbfgs_steps: Number of LBFGS refinement steps.
        seed: Random seed.

    Returns:
        FitResult with fitted parameters and diagnostics.
    """
    if not files_and_N:
        raise ValueError("files_and_N must contain at least one (csv_path, N) pair.")

    run_specs: List[RunSpec] = []
    for csv_path, n_params in files_and_N:
        if not isinstance(n_params, (int, np.integer)):
            raise TypeError(
                f"N must be an integer for path '{csv_path}', got {type(n_params)}"
            )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        run_specs.append(RunSpec(csv_path=str(csv_path), num_parameters=int(n_params)))

    N_all, D_all, y_all = build_dataset(
        pairs=run_specs,
        tokens_per_step=tokens_per_step,
        loss_column=loss_column,
        use_tokens_column=use_tokens_column,
        ignore_first_percent=ignore_first_percent,
    )

    fit = fit_parameters(
        N_all=N_all,
        D_all=D_all,
        y_all=y_all,
        huber_beta=huber_beta,
        adam_lr=adam_lr,
        adam_steps=adam_steps,
        lbfgs_steps=lbfgs_steps,
        seed=seed,
    )
    return fit


# --------------------------------------------------------------------------------------
# Minimal runnable entrypoint (default datasets)
# --------------------------------------------------------------------------------------


# (vanilla_optimal_dir / "vanilla_32d.csv", 1683000),
# (vanilla_optimal_dir / "vanilla_40d.csv", 2098000),
# (vanilla_optimal_dir / "vanilla_48d.csv", 2545000),
# (vanilla_optimal_dir / "vanilla_56d.csv", 3015000),
# (vanilla_optimal_dir / "vanilla_64d.csv", 3463000),


def get_dataset_configurations() -> dict:
    """
    Return different dataset configurations for various experiment types.
    Each configuration contains a list of (csv_path, num_parameters) pairs.
    Directly specify parameter counts - no dictionary lookups needed!
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_folder = repo_root / "experimental_data_folder"
    configurations = {}

    # 1. Optimal LR SGD scaling
    optimal_lr_sgd_dir = data_folder / "optimal_lr_sgd_scaling"
    if optimal_lr_sgd_dir.exists():
        optimal_lr_sgd_pairs = [
            (optimal_lr_sgd_dir / "optimal_lr_sgd_32d.csv", 1683000),
            (optimal_lr_sgd_dir / "optimal_lr_sgd_40d.csv", 2098000),
            (optimal_lr_sgd_dir / "optimal_lr_sgd_48d.csv", 2545000),
            (optimal_lr_sgd_dir / "optimal_lr_sgd_56d.csv", 3015000),
            (optimal_lr_sgd_dir / "optimal_lr_sgd_64d.csv", 3463000),
        ]
        configurations["optimal_lr_sgd_scaling"] = [
            (str(p), n) for p, n in optimal_lr_sgd_pairs if p.exists()
        ]

    # 2. Vanilla scaling with optimal LR
    vanilla_optimal_dir = data_folder / "vanilla_scaling_optimal_lr"
    if vanilla_optimal_dir.exists():
        vanilla_optimal_pairs = [
            (vanilla_optimal_dir / "vanilla_32d.csv", 1683000),
            (vanilla_optimal_dir / "vanilla_40d.csv", 2098000),
            (vanilla_optimal_dir / "vanilla_48d.csv", 2545000),
            (vanilla_optimal_dir / "vanilla_56d.csv", 3015000),
            (vanilla_optimal_dir / "vanilla_64d.csv", 3463000),
            (vanilla_optimal_dir / "vanilla_72d.csv", int(3.917e6)),
            (vanilla_optimal_dir / "vanilla_80d.csv", int(4.454e6)),
            (vanilla_optimal_dir / "vanilla_96d.csv", int(5.538e6)),
            (vanilla_optimal_dir / "vanilla_128d.csv", int(8.056e6)),
        ]
        configurations["vanilla_scaling_optimal_lr"] = [
            (str(p), n) for p, n in vanilla_optimal_pairs if p.exists()
        ]

    # 3. MuP scaling experiments
    mup_dir = data_folder / "mup_scaling_experiments"
    if mup_dir.exists():
        mup_pairs = [
            (mup_dir / "mup_32d.csv", 1683000),
            (mup_dir / "mup_40d.csv", 2098000),
            (mup_dir / "mup_48d.csv", 2545000),
            (mup_dir / "mup_56d.csv", 3015000),
            (mup_dir / "mup_64d.csv", 3463000),
        ]
        configurations["mup_scaling_experiments"] = [
            (str(p), n) for p, n in mup_pairs if p.exists()
        ]

    return configurations


def _default_files_and_N() -> List[Tuple[str, int]]:
    """
    Return the first available dataset configuration.
    """
    configurations = get_dataset_configurations()
    for name, pairs in configurations.items():
        if pairs:
            print(f"Using dataset: {name}")
            return pairs

    print(
        "No default files found. Please call fit_validation_loss_from_pairs(...) programmatically."
    )
    return []


def print_fit_results(dataset_name: str, fit: FitResult):
    """Print formatted results for a single dataset fit."""
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    print("Fitted parameters (L = exp(E) + exp(A) * N^alpha + exp(B) * D^beta):")
    print("Log-space parameters:")
    print(f"  E     = {fit.E:.6f}")
    print(f"  A     = {fit.A:.6f}")
    print(f"  B     = {fit.B:.6f}")
    print("Scaling exponents:")
    print(f"  alpha = {fit.alpha:.6f}")
    print(f"  beta  = {fit.beta:.6f}")
    print("Exponential form (actual scaling law coefficients):")
    print(f"  exp(E) = {fit.exp_E:.6f}  (irreducible loss)")
    print(f"  exp(A) = {fit.exp_A:.6f}  (parameter scaling coefficient)")
    print(f"  exp(B) = {fit.exp_B:.6f}  (data scaling coefficient)")
    if (fit.alpha + fit.beta) != 0:
        gamma = fit.alpha * fit.beta / (fit.alpha + fit.beta)
        a = fit.beta / (fit.alpha + fit.beta)
        b = fit.alpha / (fit.alpha + fit.beta)
        print(f"  gamma = alpha*beta/(alpha+beta) = {gamma:.6f}")
        print(f"  a = beta/(alpha+beta) = {a:.6f}")
        print(f"  b = alpha/(alpha+beta) = {b:.6f}")
    print(f"  num_points = {fit.num_points}")
    print(f"  final Huber loss = {fit.final_loss:.6f}")


def run_all_dataset_analyses(ignore_first_percent: float = 0.0):
    """Run the fitting analysis on all available datasets."""
    configurations = get_dataset_configurations()

    if not configurations:
        print(
            "No dataset configurations found. Please check your experimental_data_folder."
        )
        return

    print(f"Found {len(configurations)} dataset configurations:")
    for name in configurations.keys():
        print(f"  - {name}")

    results = {}

    for dataset_name, pairs in configurations.items():
        if not pairs:
            print(f"\nSkipping {dataset_name}: No files found")
            continue

        print(f"\nProcessing {dataset_name} with {len(pairs)} files...")

        try:
            fit = fit_validation_loss_from_pairs(
                pairs,
                loss_column="validation_loss",
                use_tokens_column=True,
                ignore_first_percent=ignore_first_percent,
            )
            results[dataset_name] = fit
            print_fit_results(dataset_name, fit)

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(
            f"{'Dataset':<30} {'Alpha':<10} {'Beta':<10} {'Exp(E)':<12} {'Points':<8}"
        )
        print("-" * 80)
        for name, fit in results.items():
            print(
                f"{name:<30} {fit.alpha:<10.4f} {fit.beta:<10.4f} {fit.exp_E:<12.6f} {fit.num_points:<8}"
            )


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    ignore_first_percent = 0.0
    if len(sys.argv) > 2:
        try:
            ignore_first_percent = float(sys.argv[2])
            if ignore_first_percent < 0.0 or ignore_first_percent >= 100.0:
                print(
                    f"Error: ignore_first_percent must be between 0.0 and 99.9, got {ignore_first_percent}"
                )
                sys.exit(1)
        except ValueError:
            print(f"Error: ignore_first_percent must be a number, got {sys.argv[2]}")
            sys.exit(1)

    # Check if user wants to run all datasets or just the default
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_dataset_analyses(ignore_first_percent=ignore_first_percent)
    else:
        # Original behavior - run on default dataset
        pairs = _default_files_and_N()
        if not pairs:
            print(
                "No default files found. Please call fit_validation_loss_from_pairs(...) programmatically."
            )
            print("Or run with --all flag to analyze all available datasets.")
            print(
                "Usage: python fit_hitchhikers_loss.py [--all] [ignore_first_percent]"
            )
            raise SystemExit(0)

        fit = fit_validation_loss_from_pairs(
            pairs,
            loss_column="validation_loss",
            use_tokens_column=True,
            ignore_first_percent=ignore_first_percent,
        )
        print_fit_results("default", fit)

# %%
