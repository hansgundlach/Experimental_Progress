from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "how-bitter-graph-cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))

import matplotlib

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the how-bitter graph figures from one parameter file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Parameter TOML file. Defaults to parameters.toml next to this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Figure output directory. Defaults to ./figures next to this script.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "numbers",
            "kaplan",
            "algorithmic-types",
            "scale-stack",
            "combined-scale",
            "reference",
            "sensitivity",
        ],
        default=None,
        help="Regenerate only selected outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        os.environ["HOW_BITTER_CONFIG"] = str(Path(args.config).expanduser().resolve())

    from parameter_loader import parameter_path
    from plots import (
        BASE_DIR,
        ensure_output_dir,
        plot_kaplan_to_chinchilla,
        save_algorithmic_progress_types,
        save_ceg_reference_combined,
        save_combined_scale_dependence_graphs,
        save_numbers,
        save_scale_dependent_stackplot,
    )
    from sensitivity_analysis import plot_sensitivity, print_exponent_gap_sensitivity_summary
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = ensure_output_dir(args.output_dir)
    selected = set(args.only or [
        "numbers",
        "kaplan",
        "algorithmic-types",
        "scale-stack",
        "combined-scale",
        "reference",
        "sensitivity",
    ])

    generated: list[Path] = []
    print(f"Using parameters: {parameter_path()}")
    print(f"Writing figures to: {output_dir}")

    if "numbers" in selected:
        numbers_path = output_dir / "numbers.txt" if args.output_dir else BASE_DIR / "numbers.txt"
        generated.append(save_numbers(numbers_path))
    if "kaplan" in selected:
        generated.extend(plot_kaplan_to_chinchilla(output_dir))
    if "algorithmic-types" in selected:
        generated.extend(save_algorithmic_progress_types(output_dir))
    if "scale-stack" in selected:
        generated.extend(save_scale_dependent_stackplot(output_dir))
    if "combined-scale" in selected:
        generated.extend(save_combined_scale_dependence_graphs(output_dir))
    if "reference" in selected:
        generated.extend(save_ceg_reference_combined(output_dir))
    if "sensitivity" in selected:
        save_path = output_dir / "sensitivity_scale_dependent_share.png"
        print()
        print_exponent_gap_sensitivity_summary(2012, 2023)
        fig, _ax = plot_sensitivity(
            np.linspace(-0.25, 0.25, 21),
            growth_year_min=2012,
            growth_year_max=2023,
            save_path=str(save_path),
        )
        plt.close(fig)
        generated.append(save_path)
        generated.append(save_path.with_suffix(".pdf"))

    print("\nGenerated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
