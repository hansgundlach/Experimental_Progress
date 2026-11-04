from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PARAMETER_PATH = Path(__file__).with_name("parameters.toml")
ENV_PARAMETER_PATH = "HOW_BITTER_CONFIG"


def parameter_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.environ.get(ENV_PARAMETER_PATH)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_PARAMETER_PATH


def load_parameters(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    resolved = parameter_path(path)
    with resolved.open("rb") as f:
        params = tomllib.load(f)

    validate_parameters(params, resolved)
    return params


def require_section(
    params: Mapping[str, Any],
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    value = params.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: missing [{key}] section")
    return value


def require_number(
    params: Mapping[str, Any],
    key: str,
    path: Path,
) -> float:
    value = params.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: `{key}` must be a number")
    return float(value)


def validate_parameters(params: Mapping[str, Any], path: Path) -> None:
    scaling_laws = require_section(params, "scaling_laws", path)
    for law_name in ("lstm", "transformer_2017", "modern_transformer"):
        law = require_section(scaling_laws, law_name, path)
        require_number(law, "A", path)
        require_number(law, "alpha", path)

    compute_frontier = require_section(params, "compute_frontier", path)
    for key in ("yearly_scaling_multiplier", "y0", "C0"):
        require_number(compute_frontier, key, path)

    chinchilla = require_section(params, "chinchilla_rebalancing", path)
    for key in (
        "coefficient",
        "term1_coefficient",
        "term1_exponent",
        "term2_coefficient",
        "term2_exponent",
        "outer_exponent",
    ):
        require_number(chinchilla, key, path)

    multipliers = require_section(params, "innovation_multipliers", path)
    for key in (
        "learned_to_rotary",
        "post_to_pre_norm",
        "sqrt_to_cosine_scheduler",
        "pre_to_rms",
        "sinusoidal_to_rotary",
        "gelu_to_swiglu",
        "mixture_of_experts",
    ):
        require_number(multipliers, key, path)

    interaction = require_section(params, "interaction_normalization", path)
    require_number(interaction, "total", path)
    if not isinstance(interaction.get("enabled"), bool):
        raise ValueError(f"{path}: `interaction_normalization.enabled` must be true or false")

    sensitivity = params.get("sensitivity", {})
    if sensitivity:
        require_number(sensitivity, "deviation", path)

    algorithmic_progress = params.get("algorithmic_progress_types", {})
    if algorithmic_progress:
        for key in (
            "baseline_A",
            "baseline_alpha",
            "steeper_A",
            "steeper_alpha",
            "shift_factor",
            "C_min",
            "C_max",
            "num_points",
        ):
            require_number(algorithmic_progress, key, path)
