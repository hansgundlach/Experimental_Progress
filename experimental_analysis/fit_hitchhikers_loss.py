import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats


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
    # Derived scaling parameters
    gamma: float  # alpha*beta/(alpha+beta)
    a: float  # beta/(alpha+beta)
    b: float  # alpha/(alpha+beta)
    # Confidence intervals (95% by default)
    gamma_ci: Tuple[float, float]  # (lower, upper) for gamma
    a_ci: Tuple[float, float]  # (lower, upper) for a
    b_ci: Tuple[float, float]  # (lower, upper) for b
    # Parameter covariance matrix (if available)
    param_cov: Optional[np.ndarray] = (
        None  # 5x5 covariance matrix for [E, A, alpha, B, beta]
    )


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
# Confidence interval utilities
# --------------------------------------------------------------------------------------


def calculate_derived_parameters(
    alpha: float, beta: float
) -> Tuple[float, float, float]:
    """Calculate gamma, a, b from alpha and beta."""
    if abs(alpha + beta) < 1e-12:  # Avoid division by zero
        return 0.0, 0.5, 0.5  # Default values when alpha + beta ≈ 0

    gamma = alpha * beta / (alpha + beta)
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)
    return gamma, a, b


def bootstrap_confidence_intervals(
    N_all: np.ndarray,
    D_all: np.ndarray,
    y_all: np.ndarray,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for gamma, a, b.

    Args:
        N_all: Parameter counts
        D_all: Token counts
        y_all: Loss values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        seed: Random seed

    Returns:
        Tuple of (gamma_ci, a_ci, b_ci) where each ci is (lower, upper)
    """
    np.random.seed(seed)
    n_points = len(y_all)

    gamma_samples = []
    a_samples = []
    b_samples = []

    alpha_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - alpha_percentile / 100) * 100

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_points, size=n_points, replace=True)
        N_boot = N_all[indices]
        D_boot = D_all[indices]
        y_boot = y_all[indices]

        try:
            # Fit on bootstrap sample - IMPORTANT: don't compute confidence intervals to avoid recursion
            result = fit_parameters(
                N_boot,
                D_boot,
                y_boot,
                seed=np.random.randint(10000),
                compute_confidence_intervals=False,
                use_chinchilla_sk_fit=False,  # Use standard fitting for bootstrap
            )
            gamma, a, b = calculate_derived_parameters(result.alpha, result.beta)

            # Only include valid results
            if np.isfinite(gamma) and np.isfinite(a) and np.isfinite(b):
                gamma_samples.append(gamma)
                a_samples.append(a)
                b_samples.append(b)

        except Exception:
            # Skip failed fits
            continue

    if len(gamma_samples) < 10:  # Need minimum number of successful fits
        print(
            f"Warning: Only {len(gamma_samples)} successful bootstrap fits out of {n_bootstrap}"
        )
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)

    # Calculate confidence intervals
    gamma_ci = (
        np.percentile(gamma_samples, alpha_percentile),
        np.percentile(gamma_samples, upper_percentile),
    )
    a_ci = (
        np.percentile(a_samples, alpha_percentile),
        np.percentile(a_samples, upper_percentile),
    )
    b_ci = (
        np.percentile(b_samples, alpha_percentile),
        np.percentile(b_samples, upper_percentile),
    )

    return (
        (float(gamma_ci[0]), float(gamma_ci[1])),
        (float(a_ci[0]), float(a_ci[1])),
        (float(b_ci[0]), float(b_ci[1])),
    )


def delta_method_confidence_intervals(
    alpha: float, beta: float, param_cov: np.ndarray, confidence_level: float = 0.95
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate confidence intervals using the delta method.

    Args:
        alpha: Fitted alpha parameter
        beta: Fitted beta parameter
        param_cov: 5x5 covariance matrix for [E, A, alpha, B, beta]
        confidence_level: Confidence level

    Returns:
        Tuple of (gamma_ci, a_ci, b_ci) where each ci is (lower, upper)
    """
    if param_cov is None or param_cov.shape != (5, 5):
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)

    # Extract covariance matrix for alpha (index 2) and beta (index 4)
    cov_alpha_alpha = param_cov[2, 2]
    cov_beta_beta = param_cov[4, 4]
    cov_alpha_beta = param_cov[2, 4]

    if abs(alpha + beta) < 1e-12:  # Avoid division by zero
        return (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)

    # Calculate gradients for delta method
    sum_ab = alpha + beta
    sum_ab_sq = sum_ab**2

    # For gamma = alpha * beta / (alpha + beta)
    d_gamma_d_alpha = beta**2 / sum_ab_sq
    d_gamma_d_beta = alpha**2 / sum_ab_sq

    # For a = beta / (alpha + beta)
    d_a_d_alpha = -beta / sum_ab_sq
    d_a_d_beta = alpha / sum_ab_sq

    # For b = alpha / (alpha + beta)
    d_b_d_alpha = beta / sum_ab_sq
    d_b_d_beta = -alpha / sum_ab_sq

    # Calculate variances using delta method
    var_gamma = (
        d_gamma_d_alpha**2 * cov_alpha_alpha
        + d_gamma_d_beta**2 * cov_beta_beta
        + 2 * d_gamma_d_alpha * d_gamma_d_beta * cov_alpha_beta
    )

    var_a = (
        d_a_d_alpha**2 * cov_alpha_alpha
        + d_a_d_beta**2 * cov_beta_beta
        + 2 * d_a_d_alpha * d_a_d_beta * cov_alpha_beta
    )

    var_b = (
        d_b_d_alpha**2 * cov_alpha_alpha
        + d_b_d_beta**2 * cov_beta_beta
        + 2 * d_b_d_alpha * d_b_d_beta * cov_alpha_beta
    )

    # Calculate confidence intervals
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    gamma, a, b = calculate_derived_parameters(alpha, beta)

    gamma_ci = (
        gamma - z_score * np.sqrt(max(0, var_gamma)),
        gamma + z_score * np.sqrt(max(0, var_gamma)),
    )
    a_ci = (a - z_score * np.sqrt(max(0, var_a)), a + z_score * np.sqrt(max(0, var_a)))
    b_ci = (b - z_score * np.sqrt(max(0, var_b)), b + z_score * np.sqrt(max(0, var_b)))

    return (
        (float(gamma_ci[0]), float(gamma_ci[1])),
        (float(a_ci[0]), float(a_ci[1])),
        (float(b_ci[0]), float(b_ci[1])),
    )


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


def chinchilla_log_loss_model(
    N: np.ndarray,
    D: np.ndarray,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    e: float,
) -> np.ndarray:
    """
    Chinchilla-style model: log L(N, D) = logsumexp(a - alpha*log(N), b - beta*log(D), e)
    Following the Chinchilla paper's approach where they minimize Huber loss of log predictions.

    Args:
        N: Number of parameters (shape: [M])
        D: Number of tokens (shape: [M])
        a: Parameter scaling coefficient (log-space)
        alpha: Scaling exponent for parameters (should be negative)
        b: Data scaling coefficient (log-space)
        beta: Scaling exponent for data (should be negative)
        e: Irreducible loss term (log-space)

    Returns:
        Predicted log loss values (shape: [M])
    """
    # Guard against numerical issues with extremely small N or D
    N_clamped = np.maximum(N, 1.0)
    D_clamped = np.maximum(D, 1.0)

    # Calculate the three terms for logsumexp
    term1 = a - alpha * np.log(N_clamped)  # Parameter scaling term
    term2 = b - beta * np.log(D_clamped)  # Data scaling term
    term3 = np.full_like(N_clamped, e)  # Irreducible loss term

    # Use logsumexp for numerical stability
    return np.logaddexp(np.logaddexp(term1, term2), term3)


def huber_loss(residuals: np.ndarray, delta: float = 0.1) -> float:
    """
    Calculate Huber loss for given residuals.

    Args:
        residuals: Array of residuals (predicted - actual)
        delta: Threshold parameter for Huber loss

    Returns:
        Huber loss value
    """
    abs_residuals = np.abs(residuals)
    quadratic_mask = abs_residuals <= delta
    linear_mask = ~quadratic_mask

    loss = np.sum(
        np.where(
            quadratic_mask, 0.5 * residuals**2, delta * (abs_residuals - 0.5 * delta)
        )
    )
    return loss


def chinchilla_sk_fit(
    N_all: np.ndarray,
    D_all: np.ndarray,
    y_all: np.ndarray,
    huber_delta: float = 0.1,
    seed: int = 42,
    confidence_level: float = 0.95,
    use_bootstrap: bool = False,
    n_bootstrap: int = 200,
    compute_confidence_intervals: bool = False,
) -> FitResult:
    """
    Fit the Chinchilla-style model using scikit-learn optimization with Huber loss.

    This follows the Chinchilla paper approach:
    - Uses logsumexp model: log L(N,D) = logsumexp(a - alpha*log(N), b - beta*log(D), e)
    - Minimizes Huber loss of log predictions vs log targets
    - Uses scikit-learn's minimize for optimization

    Args:
        N_all: Number of parameters for each data point
        D_all: Number of tokens for each data point
        y_all: Observed loss values (will be converted to log space)
        huber_delta: Delta parameter for Huber loss
        seed: Random seed for reproducibility
        confidence_level: Confidence level for intervals (default: 0.95)
        use_bootstrap: Whether to use bootstrap for confidence intervals (default: False)
        n_bootstrap: Number of bootstrap samples (default: 200)
        compute_confidence_intervals: Whether to compute confidence intervals at all (default: False)

    Returns:
        FitResult with fitted parameters, confidence intervals, and diagnostics
    """
    np.random.seed(seed)

    # Convert observed losses to log space
    log_y_all = np.log(np.maximum(y_all, 1e-10))  # Avoid log(0)

    # Initial guesses for Chinchilla model: log L = logsumexp(a - alpha*log(N), b - beta*log(D), e)
    # a_init, b_init, e_init: log-space coefficients
    log_y_mean = float(np.mean(log_y_all))
    log_y_std = float(np.std(log_y_all))

    # Better initial guesses based on the data
    a_init = log_y_mean + 0.5 * log_y_std  # Parameter scaling coefficient
    b_init = log_y_mean + 0.5 * log_y_std  # Data scaling coefficient
    e_init = log_y_mean  # Irreducible loss term
    alpha_init = -0.3  # Negative for parameter scaling (closer to typical values)
    beta_init = -0.1  # Negative for data scaling (closer to typical values)

    initial_guess = [a_init, alpha_init, b_init, beta_init, e_init]

    def objective(params):
        """Objective function to minimize: Huber loss of log predictions vs log targets."""
        a, alpha, b, beta, e = params

        try:
            # Get predicted log losses
            log_y_pred = chinchilla_log_loss_model(N_all, D_all, a, alpha, b, beta, e)

            # Calculate residuals (predicted - actual in log space)
            residuals = log_y_pred - log_y_all

            # Return Huber loss
            return huber_loss(residuals, huber_delta)

        except (ValueError, OverflowError):
            # Return large value for invalid parameters
            return 1e10

    # Set bounds for parameters
    bounds = [
        (log_y_mean - 5, log_y_mean + 5),  # a: log-space parameter coefficient
        (
            -1.0,
            -0.01,
        ),  # alpha: parameter scaling exponent (negative, but not too close to 0)
        (log_y_mean - 5, log_y_mean + 5),  # b: log-space data coefficient
        (-1.0, -0.01),  # beta: data scaling exponent (negative, but not too close to 0)
        (log_y_mean - 5, log_y_mean + 5),  # e: irreducible loss term
    ]

    try:
        # Use scipy's minimize with L-BFGS-B method
        result = minimize(
            objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
            # Use initial guess if optimization fails
            a_val, alpha_val, b_val, beta_val, e_val = initial_guess
        else:
            a_val, alpha_val, b_val, beta_val, e_val = result.x

        # Calculate final loss (Huber loss in log space)
        log_y_pred = chinchilla_log_loss_model(
            N_all, D_all, a_val, alpha_val, b_val, beta_val, e_val
        )
        residuals = log_y_pred - log_y_all
        final_loss = huber_loss(residuals, huber_delta)

        # Convert Chinchilla parameters to standard form for compatibility
        # IMPORTANT: The Chinchilla model and standard model are fundamentally different!
        # Chinchilla: log L = logsumexp(a - alpha*log(N), b - beta*log(D), e)
        # Standard: L = exp(E) + exp(A) * N^alpha + exp(B) * D^beta
        #
        # The Chinchilla model uses logsumexp (smooth max) while the standard model uses addition.
        # The parameters a, b, e in Chinchilla model are NOT directly comparable to A, B, E in standard model.
        # We map them here only for display compatibility, but they have different meanings.
        E_val = e_val
        A_val = a_val
        B_val = b_val

        # Calculate derived parameters
        gamma, a_derived, b_derived = calculate_derived_parameters(alpha_val, beta_val)

        # Calculate confidence intervals only if requested
        if compute_confidence_intervals:
            if use_bootstrap:
                # Use bootstrap method (more robust but slower)
                gamma_ci, a_ci, b_ci = bootstrap_confidence_intervals(
                    N_all, D_all, y_all, n_bootstrap, confidence_level, seed
                )
            else:
                # For Chinchilla fit, we don't have covariance matrix from scikit-learn
                # So we'll use bootstrap by default
                gamma_ci, a_ci, b_ci = bootstrap_confidence_intervals(
                    N_all, D_all, y_all, n_bootstrap, confidence_level, seed
                )
        else:
            # No confidence intervals requested
            gamma_ci, a_ci, b_ci = (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)

        fit_result = FitResult(
            E=float(E_val),
            A=float(A_val),
            alpha=float(alpha_val),
            B=float(B_val),
            beta=float(beta_val),
            huber_beta=float(huber_delta),
            num_points=int(y_all.shape[0]),
            final_loss=final_loss,
            exp_E=float(np.exp(E_val)),
            exp_A=float(np.exp(A_val)),
            exp_B=float(np.exp(B_val)),
            gamma=float(gamma),
            a=float(a_derived),
            b=float(b_derived),
            gamma_ci=gamma_ci,
            a_ci=a_ci,
            b_ci=b_ci,
            param_cov=None,  # Not available from scikit-learn optimization
        )

    except Exception as e:
        print(f"Warning: Chinchilla fitting failed with error: {e}")
        print("Falling back to initial guess values")

        # Fallback to initial guess
        gamma_init, a_init_derived, b_init_derived = calculate_derived_parameters(
            alpha_init, beta_init
        )

        fit_result = FitResult(
            E=float(e_init),
            A=float(a_init),
            alpha=float(alpha_init),
            B=float(b_init),
            beta=float(beta_init),
            huber_beta=float(huber_delta),
            num_points=int(y_all.shape[0]),
            final_loss=float("inf"),
            exp_E=float(np.exp(e_init)),
            exp_A=float(np.exp(a_init)),
            exp_B=float(np.exp(b_init)),
            gamma=float(gamma_init),
            a=float(a_init_derived),
            b=float(b_init_derived),
            gamma_ci=(np.nan, np.nan),
            a_ci=(np.nan, np.nan),
            b_ci=(np.nan, np.nan),
            param_cov=None,
        )

    return fit_result


def fit_parameters(
    N_all: np.ndarray,
    D_all: np.ndarray,
    y_all: np.ndarray,
    huber_beta: float = 0.1,
    adam_lr: float = 0.05,
    adam_steps: int = 4000,
    lbfgs_steps: int = 200,
    seed: int = 42,
    confidence_level: float = 0.95,
    use_bootstrap: bool = False,
    n_bootstrap: int = 200,
    compute_confidence_intervals: bool = False,
    use_chinchilla_sk_fit: bool = False,
) -> FitResult:
    """
    Fit the validation loss model using either scipy's curve_fit or Chinchilla-style fitting.

    Args:
        N_all: Number of parameters for each data point
        D_all: Number of tokens for each data point
        y_all: Observed loss values
        huber_beta: Not used in curve_fit (kept for compatibility)
        adam_lr: Not used in curve_fit (kept for compatibility)
        adam_steps: Not used in curve_fit (kept for compatibility)
        lbfgs_steps: Not used in curve_fit (kept for compatibility)
        seed: Random seed for reproducibility
        confidence_level: Confidence level for intervals (default: 0.95)
        use_bootstrap: Whether to use bootstrap for confidence intervals (default: False)
        n_bootstrap: Number of bootstrap samples (default: 200)
        compute_confidence_intervals: Whether to compute confidence intervals at all (default: False)
        use_chinchilla_sk_fit: Whether to use Chinchilla-style fitting with Huber loss (default: False)

    Returns:
        FitResult with fitted parameters, confidence intervals, and diagnostics
    """
    np.random.seed(seed)

    # Use Chinchilla-style fitting if requested
    if use_chinchilla_sk_fit:
        return chinchilla_sk_fit(
            N_all=N_all,
            D_all=D_all,
            y_all=y_all,
            huber_delta=huber_beta,
            seed=seed,
            confidence_level=confidence_level,
            use_bootstrap=use_bootstrap,
            n_bootstrap=n_bootstrap,
            compute_confidence_intervals=compute_confidence_intervals,
        )

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
            maxfev=2000,  # Maximum number of function evaluations (reduced for speed)
            method="trf",  # Trust Region Reflective algorithm
        )

        E_val, A_val, alpha_val, B_val, beta_val = popt

        # Calculate final loss (MSE)
        y_pred = validation_loss_model(
            N_all, D_all, E_val, A_val, alpha_val, B_val, beta_val
        )
        final_loss = float(np.mean((y_all - y_pred) ** 2))

        # Calculate derived parameters
        gamma, a, b = calculate_derived_parameters(alpha_val, beta_val)

        # Calculate confidence intervals only if requested
        if compute_confidence_intervals:
            if use_bootstrap:
                # Use bootstrap method (more robust but slower)
                gamma_ci, a_ci, b_ci = bootstrap_confidence_intervals(
                    N_all, D_all, y_all, n_bootstrap, confidence_level, seed
                )
            else:
                # Use delta method (faster but assumes normality)
                gamma_ci, a_ci, b_ci = delta_method_confidence_intervals(
                    alpha_val, beta_val, pcov, confidence_level
                )
        else:
            # No confidence intervals requested
            gamma_ci, a_ci, b_ci = (np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)

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
            gamma=float(gamma),
            a=float(a),
            b=float(b),
            gamma_ci=gamma_ci,
            a_ci=a_ci,
            b_ci=b_ci,
            param_cov=pcov,
        )

    except Exception as e:
        print(f"Warning: curve_fit failed with error: {e}")
        print("Falling back to initial guess values")

        # Fallback to initial guess
        gamma_init, a_init, b_init = calculate_derived_parameters(alpha_init, beta_init)

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
            gamma=float(gamma_init),
            a=float(a_init),
            b=float(b_init),
            gamma_ci=(np.nan, np.nan),
            a_ci=(np.nan, np.nan),
            b_ci=(np.nan, np.nan),
            param_cov=None,
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
    confidence_level: float = 0.95,
    use_bootstrap: bool = False,
    n_bootstrap: int = 200,
    compute_confidence_intervals: bool = False,
    use_chinchilla_sk_fit: bool = False,
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
        confidence_level: Confidence level for intervals (default: 0.95).
        use_bootstrap: Whether to use bootstrap for confidence intervals (default: False).
        n_bootstrap: Number of bootstrap samples (default: 200).
        compute_confidence_intervals: Whether to compute confidence intervals at all (default: False).
        use_chinchilla_sk_fit: Whether to use Chinchilla-style fitting with Huber loss (default: False).

    Returns:
        FitResult with fitted parameters, confidence intervals, and diagnostics.
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
        confidence_level=confidence_level,
        use_bootstrap=use_bootstrap,
        n_bootstrap=n_bootstrap,
        compute_confidence_intervals=compute_confidence_intervals,
        use_chinchilla_sk_fit=use_chinchilla_sk_fit,
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
    optimal_lr_sgd_dir = data_folder / "best_possible_sgd"
    if optimal_lr_sgd_dir.exists():
        optimal_lr_sgd_pairs = [
            (optimal_lr_sgd_dir / "32d_best_sgd.csv", 1683000),
            # (optimal_lr_sgd_dir / "optimal_lr_sgd_40d.csv", 2098000),
            (optimal_lr_sgd_dir / "48d_best_sgd.csv", 2545000),
            (optimal_lr_sgd_dir / "64d_best_sgd.csv", 3463000),
        ]
        configurations["best_possible_sgd"] = [
            (str(p), n) for p, n in optimal_lr_sgd_pairs if p.exists()
        ]

    # 2. Vanilla scaling with optimal LR
    # vanilla_optimal_dir = data_folder / "vanilla_scaling_optimal_lr"
    # if vanilla_optimal_dir.exists():
    #     vanilla_optimal_pairs = [
    #         # (vanilla_optimal_dir / "vanilla_16d.csv", int(0.857e6)),
    #         # (vanilla_optimal_dir / "vanilla_24d.csv", int(1.270e6)),
    #         (vanilla_optimal_dir / "vanilla_32d.csv", int(1.672e6)),
    #         (vanilla_optimal_dir / "vanilla_40d.csv", 2098000),
    #         (vanilla_optimal_dir / "vanilla_48d.csv", 2545000),
    #         (vanilla_optimal_dir / "vanilla_56d.csv", 3015000),
    #         (vanilla_optimal_dir / "vanilla_64d.csv", 3463000),
    #         # (vanilla_optimal_dir / "vanilla_72d.csv", int(3.917e6)),
    #         (vanilla_optimal_dir / "vanilla_80d.csv", int(4.454e6)),
    #         # (vanilla_optimal_dir / "vanilla_96d.csv", int(5.538e6)),
    #         # (vanilla_optimal_dir / "vanilla_128d.csv", int(8.056e6)),
    #     ]
    #     configurations["vanilla_scaling_optimal_lr"] = [
    #         (str(p), n) for p, n in vanilla_optimal_pairs if p.exists()
    #     ]

    new_scaling_dir = data_folder / "new_scaling"
    if new_scaling_dir.exists():
        new_scaling_pairs = [
            (new_scaling_dir / "32d_new_scaling.csv", 1683000),
            (new_scaling_dir / "48d_new_scaling.csv", 2545000),
            (new_scaling_dir / "64d_new_scaling.csv", 3463000),
        ]
        configurations["new_scaling"] = [
            (str(p), n) for p, n in new_scaling_pairs if p.exists()
        ]

    # no rotary scaling
    mup_dir = data_folder / "mup_scaling_experiments"
    if mup_dir.exists():
        mup_pairs = [
            (mup_dir / "vanilla_32d_no_rot.csv", 1683000),
            (mup_dir / "vanilla_40d_no_rot.csv", 2098000),
            (mup_dir / "vanilla_48d_no_rot.csv", 2545000),
            (mup_dir / "vanilla_56d_no_rot.csv", 3015000),
            (mup_dir / "vanilla_64d_no_rot.csv", 3463000),
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


def print_fit_results(dataset_name: str, fit: FitResult, is_chinchilla: bool = False):
    """Print formatted results for a single dataset fit."""
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")
    if is_chinchilla:
        print(
            "Chinchilla-style fitted parameters (log L = logsumexp(a - alpha*log(N), b - beta*log(D), e)):"
        )
        print(
            "NOTE: Parameters are mapped to standard form for display, but have different meanings!"
        )
    else:
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
        print(f"  gamma = alpha*beta/(alpha+beta) = {fit.gamma:.6f}")
        print(f"  a = beta/(alpha+beta) = {fit.a:.6f}")
        print(f"  b = alpha/(alpha+beta) = {fit.b:.6f}")

        # Display confidence intervals if available
        if not (np.isnan(fit.gamma_ci[0]) or np.isnan(fit.gamma_ci[1])):
            print("95% Confidence intervals:")
            print(f"  gamma: [{fit.gamma_ci[0]:.6f}, {fit.gamma_ci[1]:.6f}]")
            print(f"  a:     [{fit.a_ci[0]:.6f}, {fit.a_ci[1]:.6f}]")
            print(f"  b:     [{fit.b_ci[0]:.6f}, {fit.b_ci[1]:.6f}]")
    print(f"  num_points = {fit.num_points}")
    print(f"  final Huber loss = {fit.final_loss:.6f}")


def run_all_dataset_analyses(
    ignore_first_percent: float = 0.0,
    compute_confidence_intervals: bool = False,
    use_bootstrap: bool = False,
    n_bootstrap: int = 200,
    use_chinchilla_sk_fit: bool = False,
):
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
                compute_confidence_intervals=compute_confidence_intervals,
                use_bootstrap=use_bootstrap,
                n_bootstrap=n_bootstrap,
                use_chinchilla_sk_fit=use_chinchilla_sk_fit,
            )
            results[dataset_name] = fit
            print_fit_results(dataset_name, fit, is_chinchilla=use_chinchilla_sk_fit)

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

    # SET YOUR PARAMETERS HERE - both default and --all will use these exact same settings
    fit_params = {
        "loss_column": "validation_loss",
        "use_tokens_column": True,
        "ignore_first_percent": ignore_first_percent,
        "compute_confidence_intervals": True,
        "use_chinchilla_sk_fit": False,  # Change this to False for standard fitting
        "use_bootstrap": True,
        "n_bootstrap": 100,
    }

    # Check if user wants to run all datasets or just the default
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_dataset_analyses(
            ignore_first_percent=ignore_first_percent,
            compute_confidence_intervals=fit_params["compute_confidence_intervals"],
            use_bootstrap=fit_params["use_bootstrap"],
            n_bootstrap=fit_params["n_bootstrap"],
            use_chinchilla_sk_fit=fit_params["use_chinchilla_sk_fit"],
        )
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

        fit = fit_validation_loss_from_pairs(pairs, **fit_params)
        print_fit_results(
            "default", fit, is_chinchilla=fit_params["use_chinchilla_sk_fit"]
        )

# %%

# Example usage of Chinchilla-style fitting:
#
# # Standard fitting (original method)
# fit_standard = fit_validation_loss_from_pairs(
#     files_and_N,
#     use_chinchilla_sk_fit=False,
#     compute_confidence_intervals=True
# )
#
# # Chinchilla-style fitting with Huber loss
# fit_chinchilla = fit_validation_loss_from_pairs(
#     files_and_N,
#     use_chinchilla_sk_fit=True,
#     huber_beta=0.1,  # Huber loss delta parameter
#     compute_confidence_intervals=True
# )
#
# # Compare results
# print("Standard fit alpha:", fit_standard.alpha)
# print("Chinchilla fit alpha:", fit_chinchilla.alpha)
