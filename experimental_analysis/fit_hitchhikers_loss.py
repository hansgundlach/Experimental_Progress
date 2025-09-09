import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
TOKENS_PER_STEP: int = 0


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
    return grouped[["step", "tokens", "loss"]]


def build_dataset(
    pairs: Sequence[RunSpec],
    tokens_per_step: int,
    loss_column: str,
    use_tokens_column: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (csv_path, N), load and aggregate, then create arrays of N, D (tokens), and observed loss.

    Args:
        pairs: Sequence of RunSpec objects with csv_path and num_parameters
        tokens_per_step: Tokens per step (used only if use_tokens_column=False)
        loss_column: Name of the loss column
        use_tokens_column: If True, read tokens directly from 'tokens' column in CSV.
                          If False, calculate tokens as step * tokens_per_step.

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
            spec.csv_path, tokens_per_step, loss_column, use_tokens_column
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
# Model and fitting (PyTorch)
# --------------------------------------------------------------------------------------


class ValidationLossModel(torch.nn.Module):
    """
    L(N, D) = exp(E) + exp(A) * N^(-alpha) + exp(B) * D^(-beta)
    Following Hoffmann et al. (2022) form: L(f) = e^E + e^A * #params(f)^α + e^B * #toks(f)^β
    where alpha and beta are negative (scaling exponents), E, A, B are log-space parameters.
    """

    def __init__(
        self,
        E_init: float,
        A_init: float,
        alpha_init: float,
        B_init: float,
        beta_init: float,
    ):
        super().__init__()
        # All parameters are unconstrained in log space
        # E, A, B will be exponentiated, alpha and beta are direct scaling exponents
        self.E = torch.nn.Parameter(torch.tensor(float(E_init), dtype=torch.float64))
        self.A = torch.nn.Parameter(torch.tensor(float(A_init), dtype=torch.float64))
        self.alpha = torch.nn.Parameter(
            torch.tensor(float(alpha_init), dtype=torch.float64)
        )
        self.B = torch.nn.Parameter(torch.tensor(float(B_init), dtype=torch.float64))
        self.beta = torch.nn.Parameter(
            torch.tensor(float(beta_init), dtype=torch.float64)
        )

    def forward(self, N: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        # Compute L = exp(E) + exp(A) * N^alpha + exp(B) * D^beta
        # Guard against numerical issues with extremely small N or D using clamp
        N_clamped = torch.clamp(N, min=1.0)
        D_clamped = torch.clamp(D, min=1.0)
        return (
            torch.exp(self.E)
            + torch.exp(self.A) * torch.pow(N_clamped, self.alpha)
            + torch.exp(self.B) * torch.pow(D_clamped, self.beta)
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
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # Tensors
    N_t = torch.from_numpy(N_all.astype(np.float64)).to(device)
    D_t = torch.from_numpy(D_all.astype(np.float64)).to(device)
    y_t = torch.from_numpy(y_all.astype(np.float64)).to(device)

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

    model = ValidationLossModel(
        E_init=E_init,
        A_init=A_init,
        alpha_init=alpha_init,
        B_init=B_init,
        beta_init=beta_init,
    ).to(device)

    # Robust loss
    huber = torch.nn.SmoothL1Loss(beta=huber_beta, reduction="mean")

    # Phase 1: Adam for global exploration
    opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
    best_state = None
    best_loss = float("inf")

    for step in range(adam_steps):
        opt.zero_grad(set_to_none=True)
        pred = model(N_t, D_t)
        loss = huber(pred, y_t)
        # Encourage exp(E) to be near the lower envelope (weak regularization)
        reg = 1e-6 * (torch.exp(model.E) - y_t.min()).abs()
        total = loss + reg
        total.backward()
        opt.step()

        if total.item() < best_loss:
            best_loss = float(total.item())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Light cosine LR schedule
        if (step + 1) % 1000 == 0:
            for g in opt.param_groups:
                g["lr"] = max(1e-3, g["lr"] * 0.2)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2: LBFGS refine
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )

    def closure():
        opt_lbfgs.zero_grad(set_to_none=True)
        pred_lb = model(N_t, D_t)
        loss_lb = huber(pred_lb, y_t)
        loss_lb.backward()
        return loss_lb

    try:
        opt_lbfgs.step(closure)
    except RuntimeError:
        # In case of numerical issues, skip LBFGS refinement
        pass

    # Final metrics
    with torch.no_grad():
        pred_final = model(N_t, D_t)
        final_loss = float(huber(pred_final, y_t).item())

        E_val = float(model.E.item())
        A_val = float(model.A.item())
        B_val = float(model.B.item())

        result = FitResult(
            E=E_val,
            A=A_val,
            alpha=float(model.alpha.item()),
            B=B_val,
            beta=float(model.beta.item()),
            huber_beta=float(huber_beta),
            num_points=int(y_all.shape[0]),
            final_loss=final_loss,
            exp_E=float(np.exp(E_val)),
            exp_A=float(np.exp(A_val)),
            exp_B=float(np.exp(B_val)),
        )
    return result


# --------------------------------------------------------------------------------------
# Public API (programmatic use; no CLI)
# --------------------------------------------------------------------------------------


def fit_validation_loss_from_pairs(
    files_and_N: Sequence[Tuple[str, int]],
    *,
    tokens_per_step: int = TOKENS_PER_STEP,
    loss_column: str = "validation_loss",
    use_tokens_column: bool = True,
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
        tokens_per_step: Tokens per training step (used only if use_tokens_column=False, default: 65500).
        loss_column: Column name to use as target loss (default: 'validation_loss').
        use_tokens_column: If True, read tokens directly from 'tokens' column in CSV.
                          If False, calculate tokens as step * tokens_per_step (default: True).
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


def _default_files_and_N() -> List[Tuple[str, int]]:
    """
    Try Hidden_Dim_Scaling first (user-provided example); if none exist, fall back
    to muP_scaling_experiments examples.
    """
    repo_root = Path(__file__).resolve().parents[1]

    # Option 1: Hidden_Dim_Scaling (from user's example)
    # hds = repo_root / "experimental_data_folder" / "Hidden_Dim_Scaling"
    # hidden_dim_pairs = [
    #     (hds / "16d_123.csv", 810256),
    #     (hds / "24d_123.csv", 1219992),
    #     (hds / "32d_123.csv", 1632800),
    #     (hds / "64d_123.csv", 3413056),
    #     (hds / "96d_123.csv", 5488224),
    # ]
    # existing_hds = [(str(p), n) for p, n in hidden_dim_pairs if p.exists()]
    # if existing_hds:
    #     return existing_hds

    # Option 2: muP_scaling_experiments (fallback)
    # mup = repo_root / "experimental_data_folder" / "muP_scaling_experiments"
    # mup_pairs = [
    #     (mup / "16d_mup_sgd.csv", 1665073),
    #     (mup / "24d_mup_sgd.csv", 2484313),
    #     (mup / "32d_mup_sgd.csv", 3304881),
    #     (mup / "48d_mup_sgd.csv", 4988113),
    #     (mup / "64d_mup_sgd.csv", 6783185),
    #     (mup / "80d_2_mup_sgd.csv", 8636417),
    #     (mup / "96d_2_mup_sgd.csv", 10706353),
    #     (mup / "128d_2_mup_sgd.csv", 15295569),
    # ]

    mup = repo_root / "experimental_data_folder" / "generated_experiments_v100"

    mup_pairs = [
        (mup / "32d_test_experiment.csv", int(1666e3)),
        (mup / "40d_test_experiment.csv", int(2546e3)),
        (mup / "64d_test_experiment.csv", int(3482e3)),
    ]

    existing_mup = [(str(p), n) for p, n in mup_pairs if p.exists()]
    return existing_mup


if __name__ == "__main__":
    pairs = _default_files_and_N()
    if not pairs:
        print(
            "No default files found. Please call fit_validation_loss_from_pairs(...) programmatically."
        )
        raise SystemExit(0)

    fit = fit_validation_loss_from_pairs(
        pairs,
        loss_column="validation_loss",
        use_tokens_column=True,
    )
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
        print(
            f"  alpha*beta/(alpha+beta) = {fit.alpha * fit.beta / (fit.alpha + fit.beta):.6f}"
        )
    print(f"  num_points = {fit.num_points}")
    print(f"  final Huber loss = {fit.final_loss:.6f}")
