"""
Sensitivity analysis for the conclusion:
   "Proportion of algorithmic progress due to scale-dependent changes"

The scale-dependent multiplier combines Kaplan (LSTM -> 2017 Transformer)
and Chinchilla rebalancing. Both depend on the gap between the LSTM and
Modern-Transformer scaling exponents (alpha_TM - alpha_L).

This script sweeps that exponent difference as a +/- percentage of its
current value and plots the resulting scale-dependent share.

See `constants.py`:
    alpha_L_new = -deviation * (alpha_TM - alpha_L) + alpha_L
    =>  (alpha_TM - alpha_L_new) = (1 + deviation) * (alpha_TM - alpha_L)

So deviation = +0.05  -> exponent difference 5% larger
   deviation = -0.05  -> exponent difference 5% smaller
"""

import numpy as np
import matplotlib.pyplot as plt

import constants
import utils
from utils import compute_ceg_statistics, ytc


ORIG_ALPHA_L = constants.alpha_L
ORIG_ALPHA_TM = constants.alpha_TM
ORIG_DIFF = ORIG_ALPHA_TM - ORIG_ALPHA_L


def set_exponent_deviation(deviation):
    new_alpha_L = -deviation * ORIG_DIFF + ORIG_ALPHA_L
    constants.alpha_L = new_alpha_L
    utils.alpha_L = new_alpha_L
    return new_alpha_L


def reset_alpha_L():
    constants.alpha_L = ORIG_ALPHA_L
    utils.alpha_L = ORIG_ALPHA_L


def scale_dependent_share(growth_year_min, growth_year_max):
    stats = compute_ceg_statistics(growth_year_min, growth_year_max)
    return (
        np.log(stats['scale_dependent_growth'])
        / np.log(stats['total_cumulative_growth'])
        * 100
    )


def sweep(deviations, growth_year_min, growth_year_max):
    shares = []
    for d in deviations:
        set_exponent_deviation(d)
        shares.append(scale_dependent_share(growth_year_min, growth_year_max))
    reset_alpha_L()
    return np.array(shares)


ICML_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def plot_sensitivity(
    deviations,
    growth_year_min=2012,
    growth_year_max=2023,
    save_path="figures/sensitivity_scale_dependent_share.png",
):
    with plt.rc_context(ICML_RC_PARAMS):
        # ICML single-column width ~3.25"; pick a short aspect for dense layouts.
        fig, ax = plt.subplots(figsize=(3.25, 2.3))

        shares = sweep(deviations, growth_year_min, growth_year_max)
        baseline = scale_dependent_share(growth_year_min, growth_year_max)

        color = plt.cm.viridis_r(0.55)

        ax.plot(
            deviations * 100,
            shares,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
        )

        ax.axvline(0, color="0.4", linestyle="--", linewidth=0.6)
        ax.axhline(baseline, color="0.4", linestyle=":", linewidth=0.6)

        ax.annotate(
            f"baseline {baseline:.1f}%",
            xy=(0, baseline),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=6.5,
            color="0.25",
        )

        ax.set_xlabel(
            r"Change in $\alpha_{TM}-\alpha_L$ (% of current)"
        )
        ax.set_ylabel("Scale-dependent share (%)")
        ax.set_title(
            f"Sensitivity of scale-dependent share "
            f"({growth_year_min}–{growth_year_max})"
        )

        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02)

        plt.tight_layout(pad=0.3)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

        pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")

        print(f"Saved figure to {save_path} and {pdf_path}")
        return fig, ax


if __name__ == "__main__":
    deviations = np.linspace(-0.25, 0.25, 21)
    y0, y1 = 2012, 2023

    print(
        f"Baseline exponents: alpha_TM={ORIG_ALPHA_TM}, "
        f"alpha_L={ORIG_ALPHA_L}, gap={ORIG_DIFF:+.4f}\n"
    )
    print(f"--- {y0} to {y1} ---")
    print(f"{'deviation':>10}  {'new alpha_L':>12}  {'new gap':>10}  "
          f"{'share (%)':>10}")
    for d in [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        new_a = set_exponent_deviation(d)
        share = scale_dependent_share(y0, y1)
        new_gap = ORIG_ALPHA_TM - new_a
        print(f"{d*100:>+9.1f}%  {new_a:>12.5f}  "
              f"{new_gap:>+10.4f}  {share:>10.2f}")
    reset_alpha_L()
    print()

    plot_sensitivity(deviations, y0, y1)
