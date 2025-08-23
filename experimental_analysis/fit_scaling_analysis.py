"""
Fit scaling law using the Chinchilla-style objective on log-losses:
min over a,b,e,alpha,beta of Huber_delta( LSE(a + alpha*log N, b + beta*log D, e) - log L ),
then set A=exp(a), B=exp(b), E=exp(e). We use L-BFGS with multi-start grid.

CSV expectations (defaults can be overridden via CLI):
- steps column: "step"
- validation loss column: "validation_loss"

Tokens conversion: N = steps * TOKENS_PER_STEP.
Edit TOKENS_PER_STEP below as desired, or override with --tokens-per-step.
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from pathlib import Path


# Set default dtype for better numerical accuracy
torch.set_default_dtype(torch.float64)


# EDIT THIS: conversion from steps to tokens (can also be overridden via CLI)
TOKENS_PER_STEP_DEFAULT: float = 65536

# Optional: set to a float (e.g., 1.7) to FIX E during optimization
FIX_E_TO: Optional[float] = 1.7

# Safety: limit number of grid initializations (set None for full grid)
MAX_GRID_EVALS: Optional[int] = 1000


@dataclass
class FitResult:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    huber_delta: float
    final_loss: float
    num_points: int


def convert_steps_to_tokens(steps: np.ndarray, tokens_per_step: float) -> np.ndarray:
    return steps.astype(np.float64) * float(tokens_per_step)


def load_points_from_csv(
    csv_paths: List[str],
    steps_col: str,
    loss_col: str,
    n_values: List[float],
    d_col: Optional[str] = None,
    d_default: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps_list: List[np.ndarray] = []
    loss_list: List[np.ndarray] = []
    d_list: List[np.ndarray] = []
    n_list: List[np.ndarray] = []
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        if steps_col not in df.columns:
            raise ValueError(
                f"Column '{steps_col}' not found in {path}. Columns: {list(df.columns)}"
            )
        if loss_col not in df.columns:
            raise ValueError(
                f"Column '{loss_col}' not found in {path}. Columns: {list(df.columns)}"
            )
        s = df[steps_col].to_numpy()
        y = df[loss_col].to_numpy()
        if d_col is not None and d_col in df.columns:
            d_vals = df[d_col].to_numpy()
        else:
            # Fallback: D = step * TOKENS_PER_STEP_DEFAULT
            d_vals = s.astype(np.float64) * float(TOKENS_PER_STEP_DEFAULT)

        # N is constant per file
        n_vals = np.full_like(s, fill_value=float(n_values[i]), dtype=np.float64)

        # Filter out non-finite and non-positive values
        mask = (
            np.isfinite(s)
            & np.isfinite(y)
            & np.isfinite(d_vals)
            & np.isfinite(n_vals)
            & (s > 0)
            & (y > 0)
            & (d_vals > 0)
            & (n_vals > 0)
        )
        s = s[mask]
        y = y[mask]
        d_vals = d_vals[mask]
        n_vals = n_vals[mask]
        steps_list.append(s.astype(np.float64))
        loss_list.append(y.astype(np.float64))
        d_list.append(d_vals.astype(np.float64))
        n_list.append(n_vals.astype(np.float64))
    if not steps_list:
        raise ValueError("No valid data points found across provided CSVs.")
    steps_all = np.concatenate(steps_list, axis=0)
    loss_all = np.concatenate(loss_list, axis=0)
    d_all = np.concatenate(d_list, axis=0)
    n_all = np.concatenate(n_list, axis=0)
    return steps_all, loss_all, d_all, n_all


def predict_loss(
    tokens: torch.Tensor,
    E: torch.Tensor,
    log_A: torch.Tensor,
    log_alpha: torch.Tensor,
    log_B: torch.Tensor,
    log_beta: torch.Tensor,
) -> torch.Tensor:
    # Ensure positivity of A, a, B, b via exp parameterization
    A = torch.exp(log_A)
    alpha = torch.exp(log_alpha)
    B = torch.exp(log_B)
    beta = torch.exp(log_beta)
    return E + A / torch.pow(tokens, alpha) + B / torch.pow(tokens, beta)


def smooth_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, delta: float
) -> torch.Tensor:
    # Huber loss (SmoothL1) with custom delta
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(
        abs_diff, torch.tensor(delta, dtype=pred.dtype, device=pred.device)
    )
    # 0.5 * x^2 region for |x| <= delta, and linear region otherwise
    linear = abs_diff - quadratic
    return (0.5 * quadratic * quadratic + delta * linear).mean()


def initialize_parameters(
    tokens: np.ndarray, losses: np.ndarray
) -> Tuple[torch.nn.Parameter, ...]:
    # Heuristics: E ~ average of top-10% largest tokens losses
    num_points = len(tokens)
    if num_points == 0:
        raise ValueError("No points to initialize from.")
    sort_idx = np.argsort(tokens)
    top_k = max(1, num_points // 10)
    # E0 = float(np.mean(losses[sort_idx][-top_k:]))
    E0 = 1.7
    # Initial exponents and amplitudes
    alpha0 = 0.2
    beta0 = 0.01
    # Rough scale for A and B from initial residual at smallest N
    min_tokens = float(np.min(tokens))
    residual0 = (
        float(np.mean(losses[tokens <= min_tokens * 2]) - E0)
        if np.any(tokens <= min_tokens * 2)
        else float(np.mean(losses) - E0)
    )
    residual0 = max(residual0, 1e-3)
    A0 = 0.5 * residual0 * (min_tokens**alpha0)
    B0 = 0.5 * residual0 * (min_tokens**beta0)

    E = torch.nn.Parameter(torch.tensor(E0, dtype=torch.get_default_dtype()))
    log_A = torch.nn.Parameter(
        torch.log(torch.tensor(A0, dtype=torch.get_default_dtype()))
    )
    log_alpha = torch.nn.Parameter(
        torch.log(torch.tensor(alpha0, dtype=torch.get_default_dtype()))
    )
    log_B = torch.nn.Parameter(
        torch.log(torch.tensor(B0, dtype=torch.get_default_dtype()))
    )
    log_beta = torch.nn.Parameter(
        torch.log(torch.tensor(beta0, dtype=torch.get_default_dtype()))
    )
    return E, log_A, log_alpha, log_B, log_beta


def fit_scaling_law(
    csv_paths: List[str],
    steps_col: str,
    loss_col: str,
    n_values: List[float],
    d_col: Optional[str],
    huber_delta: float,
    max_iter: int,
    fixed_E: Optional[float] = None,
) -> FitResult:
    steps, losses, dims, n_vals = load_points_from_csv(
        csv_paths, steps_col, loss_col, n_values, d_col=d_col, d_default=1.0
    )

    # Convert to tensors
    n_vals_t = torch.from_numpy(n_vals)
    losses_t = torch.from_numpy(losses)
    dims_t = torch.from_numpy(dims)
    log_n = torch.log(n_vals_t)
    log_dims = torch.log(dims_t)
    log_losses = torch.log(losses_t)

    # DEBUG: Show basic stats to ensure variation is present
    print("[DEBUG] Data stats:")
    print("  files:", len(csv_paths))
    print("  points:", len(n_vals))
    print(
        "  log_n: min={:.3f}, max={:.3f}, std={:.3f}".format(
            float(log_n.min()), float(log_n.max()), float(log_n.std())
        )
    )
    print(
        "  log_D(step): min={:.3f}, max={:.3f}, std={:.3f}".format(
            float(log_dims.min()), float(log_dims.max()), float(log_dims.std())
        )
    )
    print(
        "  log_L: min={:.3f}, max={:.3f}, std={:.3f}".format(
            float(log_losses.min()), float(log_losses.max()), float(log_losses.std())
        )
    )

    # Baseline initialization (used to scale grid-inits)
    E_base, log_A_base, log_alpha_base, log_B_base, log_beta_base = (
        initialize_parameters(n_vals, losses)
    )
    with torch.no_grad():
        base_E = float(E_base.item())
        base_A = float(torch.exp(log_A_base).item())
        base_alpha = float(torch.exp(log_alpha_base).item())
        base_B = float(torch.exp(log_B_base).item())
        base_beta = float(torch.exp(log_beta_base).item())

    # Chinchilla-style grid with priority on alpha/beta (finer sweep),
    # and inner LBFGS optimizing a,b,e for each (alpha,beta).
    # Center around literature values (~0.2-0.8) and exclude 0.0 to avoid collapse
    alpha_grid = np.arange(0.05, 0.81, 0.2)
    beta_grid = np.arange(0.05, 0.81, 0.2)

    best_loss: float = float("inf")
    best_params: Tuple[float, float, float, float, float] = (
        base_E,
        base_A,
        base_alpha,
        base_B,
        base_beta,
    )

    def optimize_given_alpha_beta(
        alpha_fixed: float,
        beta_fixed: float,
        a_seed: float,
        e_seed: float,
        b_seed: float,
    ) -> Tuple[float, float, float, float, float, float]:
        # Optimize only a,b,e with alpha,beta fixed
        if fixed_E is not None:
            E_param = torch.tensor(
                math.log(float(fixed_E)), dtype=torch.get_default_dtype()
            )
            a_param = torch.nn.Parameter(
                torch.tensor(a_seed, dtype=torch.get_default_dtype())
            )
            b_param = torch.nn.Parameter(
                torch.tensor(b_seed, dtype=torch.get_default_dtype())
            )
            params = [a_param, b_param]
        else:
            E_param = torch.nn.Parameter(
                torch.tensor(e_seed, dtype=torch.get_default_dtype())
            )
            a_param = torch.nn.Parameter(
                torch.tensor(a_seed, dtype=torch.get_default_dtype())
            )
            b_param = torch.nn.Parameter(
                torch.tensor(b_seed, dtype=torch.get_default_dtype())
            )
            params = [E_param, a_param, b_param]

        alpha_const = torch.tensor(alpha_fixed, dtype=torch.get_default_dtype())
        beta_const = torch.tensor(beta_fixed, dtype=torch.get_default_dtype())

        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            term1 = a_param - alpha_const * log_n
            term2 = b_param - beta_const * log_dims
            term3 = (
                E_param if isinstance(E_param, torch.nn.Parameter) else E_param
            ) + torch.zeros_like(log_n)
            pred_log = torch.logsumexp(
                torch.stack([term1, term2, term3], dim=-1), dim=-1
            )
            loss = smooth_l1_loss(pred_log, log_losses, huber_delta)
            loss.backward()
            return loss

        step_result = optimizer.step(closure)
        final = (
            float(step_result.detach().cpu().item())
            if isinstance(step_result, torch.Tensor)
            else float(step_result)
        )
        with torch.no_grad():
            A_final = float(torch.exp(a_param).item())
            B_final = float(torch.exp(b_param).item())
            E_final = (
                float(fixed_E)
                if fixed_E is not None
                else float(torch.exp(E_param).item())
            )
        return final, E_final, A_final, alpha_fixed, B_final, beta_fixed

    # Iterate grid
    # Seeds for a,b,e based on data
    median_logL = float(np.median(np.log(losses)))
    a_seeds = [median_logL - 1.0, median_logL, median_logL + 1.0]
    b_seeds = [median_logL - 1.0, median_logL, median_logL + 1.0]
    e_seeds = [median_logL - 0.5, median_logL, median_logL + 0.5]

    eval_count = 0
    for alpha_val in alpha_grid:
        for beta_val in beta_grid:
            for a0 in a_seeds:
                for b0 in b_seeds:
                    for e0 in e_seeds:
                        loss_val, E_v, A_v, alpha_v, B_v, beta_v = (
                            optimize_given_alpha_beta(alpha_val, beta_val, a0, e0, b0)
                        )
                        eval_count += 1
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_params = (E_v, A_v, alpha_v, B_v, beta_v)
                        if MAX_GRID_EVALS is not None and eval_count >= MAX_GRID_EVALS:
                            break
                    if MAX_GRID_EVALS is not None and eval_count >= MAX_GRID_EVALS:
                        break
                if MAX_GRID_EVALS is not None and eval_count >= MAX_GRID_EVALS:
                    break
            if MAX_GRID_EVALS is not None and eval_count >= MAX_GRID_EVALS:
                break

    E_best, A_best, alpha_best, B_best, beta_best = best_params

    # Final refinement: jointly optimize a,b,e,alpha,beta from grid best
    def refine_all_params(
        E0: float, A0: float, alpha0: float, B0: float, beta0: float
    ) -> Tuple[float, float, float, float, float, float]:
        # If fixed_E is provided to outer scope, respect it here too by not optimizing E
        if fixed_E is None:
            e_param = torch.nn.Parameter(
                torch.tensor(math.log(max(E0, 1e-8)), dtype=torch.get_default_dtype())
            )
        else:
            e_fixed = torch.tensor(
                math.log(max(float(fixed_E), 1e-8)), dtype=torch.get_default_dtype()
            )
        a_param = torch.nn.Parameter(
            torch.tensor(math.log(max(A0, 1e-12)), dtype=torch.get_default_dtype())
        )
        b_param = torch.nn.Parameter(
            torch.tensor(math.log(max(B0, 1e-12)), dtype=torch.get_default_dtype())
        )
        alpha_raw = torch.nn.Parameter(
            torch.tensor(math.sqrt(max(alpha0, 1e-6)), dtype=torch.get_default_dtype())
        )
        beta_raw = torch.nn.Parameter(
            torch.tensor(math.sqrt(max(beta0, 1e-6)), dtype=torch.get_default_dtype())
        )
        params = [a_param, b_param, alpha_raw, beta_raw]
        if fixed_E is None:
            params = [e_param] + params

        opt = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
        )

        def clos():
            opt.zero_grad(set_to_none=True)
            alpha_val = alpha_raw * alpha_raw
            beta_val = beta_raw * beta_raw
            term1 = a_param - alpha_val * log_n
            term2 = b_param - beta_val * log_dims
            term3 = (e_param if fixed_E is None else e_fixed) + torch.zeros_like(log_n)
            pred_log = torch.logsumexp(
                torch.stack([term1, term2, term3], dim=-1), dim=-1
            )
            loss = smooth_l1_loss(pred_log, log_losses, huber_delta)
            loss.backward()
            return loss

        refin = opt.step(clos)
        refin_loss = (
            float(refin.detach().cpu().item())
            if isinstance(refin, torch.Tensor)
            else float(refin)
        )
        with torch.no_grad():
            E_ref = (
                float(fixed_E)
                if fixed_E is not None
                else float(torch.exp(e_param).item())
            )
            A_ref = float(torch.exp(a_param).item())
            B_ref = float(torch.exp(b_param).item())
            alpha_ref = float((alpha_raw * alpha_raw).item())
            beta_ref = float((beta_raw * beta_raw).item())
        return refin_loss, E_ref, A_ref, alpha_ref, B_ref, beta_ref

    refin_loss, E_r, A_r, alpha_r, B_r, beta_r = refine_all_params(
        E_best, A_best, alpha_best, B_best, beta_best
    )
    if refin_loss < best_loss:
        E_best, A_best, alpha_best, B_best, beta_best = E_r, A_r, alpha_r, B_r, beta_r
        best_loss = refin_loss

    return FitResult(
        E=E_best,
        A=A_best,
        alpha=alpha_best,
        B=B_best,
        beta=beta_best,
        huber_delta=huber_delta,
        final_loss=best_loss,
        num_points=len(n_vals),
    )


# 810256 this is 16 size
# 1219992 this is 24 size
# 1632800 this is 32 size
# 2495280 this is 48 size
# 2927288 this is 56 size
# 3413056 this is 64 size
# 3867336 this is 72 size
# 4404560 this is 80 size
# 5488224 this is 96 size
# 8005760 this is 128 size


def main():
    # Fit each file individually with its own N value
    # EDIT THE N VALUES BELOW AS NEEDED
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "experimental_data_folder" / "muP_scaling_experiments"

    # files_and_N = [
    #     (data_dir / "16d_mup_sgd.csv", 1665073),
    #     (data_dir / "24d_mup_sgd.csv", 2484313),
    #     (data_dir / "32d_mup_sgd.csv", 3304881),
    #     (data_dir / "48d_mup_sgd.csv", 4988113),
    #     (data_dir / "64d_mup_sgd.csv", 6783185),
    #     (data_dir / "80d_2_mup_sgd.csv", 8636417),
    #     (data_dir / "96d_2_mup_sgd.csv", 10706353),
    #     (data_dir / "128d_2_mup_sgd.csv", 15295569),
    # ]

    # {'hidden_dim': 96, 'num_layers

    data_dir = repo_root / "experimental_data_folder" / "Hidden_Dim_Scaling"

    files_and_N = [
        (data_dir / "16d_123.csv", 810256),
        (data_dir / "24d_123.csv", 1219992),
        (data_dir / "32d_123.csv", 1632800),
        (data_dir / "64d_123.csv", 3413056),
        (data_dir / "96d_123.csv", 5488224),
    ]

    # Filter existing files
    existing_files = []
    existing_N = []
    for csv_file, N_value in files_and_N:
        if csv_file.exists():
            existing_files.append(str(csv_file))
            existing_N.append(N_value)
        else:
            print(f"Skipping {csv_file.name} (file not found)")

    if not existing_files:
        print("No valid files found!")
        return

    # Fit all files together with one set of parameters
    result = fit_scaling_law(
        csv_paths=existing_files,
        steps_col="step",
        loss_col="validation_loss",
        n_values=existing_N,
        d_col=None,  # D will be set to step automatically
        huber_delta=1e-3,
        max_iter=500,
        fixed_E=FIX_E_TO,
    )

    print(f"\n=== GLOBAL FIT ACROSS ALL FILES ===")
    print(f"Files used: {len(existing_files)}")
    print(f"N values: {existing_N}")
    print(f"E = {result.E:.8f}")
    print(f"A = {result.A:.8f}")
    print(f"alpha = {result.alpha:.8f}")
    print(f"B = {result.B:.8f}")
    print(f"beta = {result.beta:.8f}")
    if (result.alpha + result.beta) > 0:
        print(
            f"alpha*beta/(alpha+beta) = {result.alpha * result.beta / (result.alpha + result.beta):.8f}"
        )
    print(f"Final objective: {result.final_loss:.8f}")
    print(f"Total points used: {result.num_points}")


if __name__ == "__main__":
    main()
