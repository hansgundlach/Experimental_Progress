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


@dataclass
class RunSpec:
    csv_path: str
    num_parameters: int


@dataclass
class FitResult:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    huber_beta: float
    num_points: int
    final_loss: float


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def read_and_aggregate_csv(
    csv_path: str, tokens_per_step: int, loss_column: str
) -> pd.DataFrame:
    """
    Reads a CSV with columns including 'step' and the given loss column, averages over duplicate steps,
    filters out step == 0 (since D=0 tokens is not valid for the model), and returns a DataFrame with
    columns: step, tokens (D), loss.
    """
    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'step' column.")
    if loss_column not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a '{loss_column}' column.")

    # Group by step to average repeated evaluations/log lines
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
    pairs: Sequence[RunSpec], tokens_per_step: int, loss_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (csv_path, N), load and aggregate, then create arrays of N, D (tokens), and observed loss.
    Returns:
      - N_all: shape [M]
      - D_all: shape [M]
      - y_all: shape [M]
    """
    N_list: List[float] = []
    D_list: List[float] = []
    y_list: List[float] = []

    for spec in pairs:
        agg = read_and_aggregate_csv(spec.csv_path, tokens_per_step, loss_column)
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
    L(N, D) = E + A / N^alpha + B / D^beta
    Enforces positivity for A, alpha, B, beta via softplus. E is free.
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
        # raw parameters (unconstrained)
        self.E = torch.nn.Parameter(torch.tensor(float(E_init), dtype=torch.float64))
        self.A_raw = torch.nn.Parameter(
            torch.tensor(math.log(math.expm1(max(A_init, 1e-6))), dtype=torch.float64)
        )
        self.alpha_raw = torch.nn.Parameter(
            torch.tensor(
                math.log(math.expm1(max(alpha_init, 1e-6))), dtype=torch.float64
            )
        )
        self.B_raw = torch.nn.Parameter(
            torch.tensor(math.log(math.expm1(max(B_init, 1e-6))), dtype=torch.float64)
        )
        self.beta_raw = torch.nn.Parameter(
            torch.tensor(
                math.log(math.expm1(max(beta_init, 1e-6))), dtype=torch.float64
            )
        )

        self.softplus = torch.nn.Softplus()

    @property
    def A(self) -> torch.Tensor:
        return self.softplus(self.A_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return self.softplus(self.alpha_raw)

    @property
    def B(self) -> torch.Tensor:
        return self.softplus(self.B_raw)

    @property
    def beta(self) -> torch.Tensor:
        return self.softplus(self.beta_raw)

    def forward(self, N: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        # Compute L = E + A / N^alpha + B / D^beta
        # Guard against numerical issues with extremely small N or D using clamp
        N_clamped = torch.clamp(N, min=1.0)
        D_clamped = torch.clamp(D, min=1.0)
        return (
            self.E
            + self.A / torch.pow(N_clamped, self.alpha)
            + self.B / torch.pow(D_clamped, self.beta)
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

    # Initial guesses
    E_init = float(np.percentile(y_all, 10))
    A_init = max(float(np.percentile(y_all - E_init, 90)), 1e-3)
    alpha_init = 0.5
    B_init = max(float(np.percentile(y_all - E_init, 90)), 1e-3)
    beta_init = 0.5

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
        # Encourage E to be near the lower envelope (weak regularization)
        reg = 1e-6 * (model.E - y_t.min()).abs()
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
        result = FitResult(
            E=float(model.E.item()),
            A=float(model.A.item()),
            alpha=float(model.alpha.item()),
            B=float(model.B.item()),
            beta=float(model.beta.item()),
            huber_beta=float(huber_beta),
            num_points=int(y_all.shape[0]),
            final_loss=final_loss,
        )
    return result


# --------------------------------------------------------------------------------------
# Public API (programmatic use; no CLI)
# --------------------------------------------------------------------------------------


def fit_validation_loss_from_pairs(
    files_and_N: Sequence[Tuple[str, int]],
    *,
    tokens_per_step: int = 65500,
    loss_column: str = "validation_loss",
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
        tokens_per_step: Tokens per training step (default: 65500).
        loss_column: Column name to use as target loss (default: 'validation_loss').
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
        pairs=run_specs, tokens_per_step=tokens_per_step, loss_column=loss_column
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
    mup = repo_root / "experimental_data_folder" / "muP_scaling_experiments"
    mup_pairs = [
        (mup / "16d_mup_sgd.csv", 1665073),
        (mup / "24d_mup_sgd.csv", 2484313),
        (mup / "32d_mup_sgd.csv", 3304881),
        (mup / "48d_mup_sgd.csv", 4988113),
        (mup / "64d_mup_sgd.csv", 6783185),
        (mup / "80d_2_mup_sgd.csv", 8636417),
        (mup / "96d_2_mup_sgd.csv", 10706353),
        (mup / "128d_2_mup_sgd.csv", 15295569),
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
        pairs, tokens_per_step=65500, loss_column="validation_loss"
    )
    print("Fitted parameters (L = E + A/N^alpha + B/D^beta):")
    print(f"  E     = {fit.E:.6f}")
    print(f"  A     = {fit.A:.6f}")
    print(f"  alpha = {fit.alpha:.6f}")
    print(f"  B     = {fit.B:.6f}")
    print(f"  beta  = {fit.beta:.6f}")
    if (fit.alpha + fit.beta) > 0:
        print(
            f"  alpha*beta/(alpha+beta) = {fit.alpha * fit.beta / (fit.alpha + fit.beta):.6f}"
        )
    print(f"  num_points = {fit.num_points}")
    print(f"  final Huber loss = {fit.final_loss:.6f}")
