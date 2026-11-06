from __future__ import annotations

import math
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from utils import *


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURE_DIR = BASE_DIR / "figures"


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    out = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def close_and_return(fig, *paths: Path) -> list[Path]:
    plt.close(fig)
    return [Path(p) for p in paths]


def save_numbers(output_path: str | Path | None = None) -> Path:
    output_path = Path(output_path) if output_path is not None else BASE_DIR / "numbers.txt"
    blocks = [
        (2012, 2023),
        (2013, 2025),
    ]
    output = ""

    for growth_year_min, growth_year_max in blocks:
        stats = compute_ceg_statistics(growth_year_min, growth_year_max)
        output += format_stats_block(stats)
        if (growth_year_min, growth_year_max) == (2012, 2023):
            output += f"{2**(12/8):>12.2f}x  Ho et al annual growth rate (from {2012} to {2023})\n"
        output += "\n"

    stats = compute_ceg_statistics(2013, 2017.01)
    output += (
        f"{stats['total_transformer_growth']:>12.2f}x  Total transformer growth "
        f"(from {stats['growth_year_min']} to {stats['growth_year_max']})\n"
    )
    C_2017 = ytc(2017)
    lstm_to_kaplan_2017 = lstm_to_2017_transformer_multiplier(C_2017, kaplan_divide=False)
    output += (
        f"{lstm_to_kaplan_2017:>12.2f}x  LSTM to Kaplan Transformer efficiency gain "
        f"at 2017 compute ({C_2017:.3g} FLOPs)\n"
    )
    output += "\n"
    output += format_ho_gap_block(2012, 2023)
    output += "\n"
    output += format_exponent_gap_sensitivity_block(2012, 2023)

    output_path.write_text(output)
    return output_path


def format_ho_gap_block(growth_year_min: float, growth_year_max: float) -> str:
    stats = compute_ceg_statistics(growth_year_min, growth_year_max)
    ho_total_ceg = 22000
    gap = ho_total_ceg / stats["total_cumulative_growth"]
    return (
        f"{gap:>12.2f}x  Gap to Ho et al. = 22,000 / total CEG "
        f"in {growth_year_max:g} ({stats['total_cumulative_growth']:.2f}x)\n"
    )


def format_exponent_gap_sensitivity_block(growth_year_min: float, growth_year_max: float) -> str:
    from sensitivity_analysis import exponent_gap_sensitivity_rows

    labels = {
        -0.25: "Scale-dependent proportion with exponent gap 25% smaller",
        0.0: "Scale-dependent proportion with baseline exponent gap",
        0.25: "Scale-dependent proportion with exponent gap 25% larger",
    }
    output = f"Exponent-gap sensitivity ({growth_year_min:g} to {growth_year_max:g}):\n"
    for row in exponent_gap_sensitivity_rows(growth_year_min, growth_year_max):
        output += f"{row['share']:>12.2f}%  {labels[row['deviation']]}\n"
    return output


def format_stats_block(stats: dict[str, float]) -> str:
    growth_year_min = stats["growth_year_min"]
    growth_year_max = stats["growth_year_max"]
    output = ""
    output += f"{stats['total_cumulative_growth']:>12.2f}x  Final cumulative multiplier (from {growth_year_min} to {growth_year_max})\n"
    output += f"{22000 / stats['total_cumulative_growth']:>12.2f}x  Gap from Ho et al. (from {growth_year_min} to {growth_year_max})\n"
    output += f"{(np.log(stats['scale_dependent_growth']) / np.log(stats['total_cumulative_growth']) * 100):>12.2f}%  Proportion from scale-dependent (from {growth_year_min} to {growth_year_max})\n"
    output += f"{stats['total_transformer_growth']:>12.2f}x  Total multiplier from from transformer (from {growth_year_min} to {growth_year_max})\n"
    output += f"{(np.log(stats['total_transformer_growth']) / np.log(stats['total_cumulative_growth']) * 100):>12.2f}%  Proportion from from transformer (from {growth_year_min} to {growth_year_max})\n"
    output += f"{stats['total_chinchilla_growth']:>12.2f}x  Total multiplier from from Chinchilla (from {growth_year_min} to {growth_year_max})\n"
    output += f"{(np.log(stats['total_chinchilla_growth']) / np.log(stats['total_cumulative_growth']) * 100):>12.2f}%  Proportion from from Chinchilla (from {growth_year_min} to {growth_year_max})\n"
    output += f"{stats['scale_dependent_growth']:>12.2f}x  Scale-dependent multiplier (from {growth_year_min} to {growth_year_max})\n"
    output += f"{(np.log(stats['scale_invariant_growth']) / np.log(stats['total_cumulative_growth']) * 100):>12.2f}%  Proportion from scale-invariant (from {growth_year_min} to {growth_year_max})\n"
    output += f"{stats['scale_invariant_growth']:>12.2f}x  Scale-invariant multiplier (from {growth_year_min} to {growth_year_max})\n"
    output += f"{(stats['total_cumulative_growth'] / stats['initial_cumulative_growth']) ** (1/(growth_year_max-growth_year_min)):>12.4f}x  Rate of growth (from {growth_year_min} to {growth_year_max})\n"
    output += f"               Compute range for analysis: {ytc(growth_year_min):.3g}, {ytc(growth_year_max):.3g}\n"
    return output


def plot_kaplan_to_chinchilla(output_dir: str | Path | None = None) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    kaplan_color = INNOVATIONS["kaplan"]["color"]
    chinchilla_color = INNOVATIONS["chinchilla"]["color"]

    E, A, B = 1.69, 406.4, 410.7
    alpha, beta = 0.34, 0.28
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    a_K, b_K = 0.73, 0.27
    a_C, b_C = 0.46, 0.54
    kaplan_n_constant = 3.6331917579105387e-06

    def chinchilla_loss(N, D, irr_loss=False):
        return (0 if not irr_loss else E) + A / (N ** alpha) + B / (D ** beta)

    def n_opt(C, paper="kaplan"):
        if paper == "kaplan":
            return kaplan_n_constant * C**a_K
        if paper == "chinchilla":
            return G * (C / 6) ** a_C
        raise ValueError(f"Unknown paper: {paper}")

    def d_opt(C, paper="kaplan"):
        if paper == "kaplan":
            return 1 / (6 * kaplan_n_constant) * C**b_K
        if paper == "chinchilla":
            return (1 / G) * (C / 6) ** b_C
        raise ValueError(f"Unknown paper: {paper}")

    Cs = np.array([10**i for i in range(15, 30)], dtype=float)
    kaplan_losses = np.array([chinchilla_loss(n_opt(C, "kaplan"), d_opt(C, "kaplan")) for C in Cs])
    chinchilla_losses = np.array([chinchilla_loss(n_opt(C, "chinchilla"), d_opt(C, "chinchilla")) for C in Cs])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Cs, chinchilla_losses, linewidth=2.5, color=chinchilla_color)
    ax.plot(Cs, kaplan_losses, linewidth=2.5, color=kaplan_color)

    max_c = Cs[-1]
    for final_loss, color, label in (
        (chinchilla_losses[-1], chinchilla_color, "Chinchilla Scaling"),
        (kaplan_losses[-1], kaplan_color, "Kaplan Scaling"),
    ):
        ax.plot([max_c * 0.995, max_c], [final_loss, final_loss], color=color, linewidth=1.5, alpha=0.8, clip_on=False)
        ax.plot([max_c, max_c * 1.5], [final_loss, final_loss], color=color, linewidth=1, alpha=0.7, clip_on=False)
        ax.text(max_c * 1.6, final_loss, label, color=color, fontsize=12, va="center", fontweight="bold", clip_on=False)

    ax.set_ylabel("Validation Loss - Irreducible", fontsize=14)
    ax.set_xlabel("Compute (FLOPs)", fontsize=14)
    ax.set_title(
        "Switching from Kaplan to Chinchilla Scaling\nhas Scale Dependent Improvements",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(float(Cs[0]), float(Cs[-1]))
    ax.set_ylim(1e-2, 1e2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.5, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    png_path = output_dir / "kaplan_to_chinchilla.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    return close_and_return(fig, png_path)


def save_algorithmic_progress_types(output_dir: str | Path | None = None) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    params = PARAMETERS.get("algorithmic_progress_types", {})
    A0 = params.get("baseline_A", 1.0)
    alpha0 = params.get("baseline_alpha", 0.35)
    A1 = params.get("steeper_A", 1.0)
    alpha1 = params.get("steeper_alpha", 0.55)
    shift_factor = params.get("shift_factor", 0.6)
    c_min = params.get("C_min", 1.0)
    c_max = params.get("C_max", 1.0e4)
    num_points = int(params.get("num_points", 400))

    def power_law(C, A, alpha):
        return A * C ** (-alpha)

    C = np.logspace(np.log10(c_min), np.log10(c_max), num_points)
    ctx = power_law(C, A0, alpha0)
    steeper = power_law(C, A1, alpha1)
    shifted = power_law(C, A1 * shift_factor, alpha1)

    pal = sns.color_palette("viridis", 3)
    col_ctx = "0.6"
    col_steeper = pal[2]
    col_shift = pal[1]

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        ax.loglog(C, ctx, "--", lw=2, color=col_ctx, alpha=0.9, label="Original baseline (context)")
        ax.loglog(C, steeper, "-", lw=3, color=col_steeper, label="Exponent Change")
        ax.loglog(C, shifted, "--", lw=3, color=col_shift, label="Shift")

        ax.set_title("Two Types of Algorithmic Progress", weight="bold", pad=14)
        ax.set_xlabel("Compute (FLOP, log scale)")
        ax.set_ylabel("Loss - Irreducible Loss (log scale)")
        ax.grid(which="both", linestyle=":", linewidth=0.8, alpha=0.7)

        x_s = C[int(len(C) * 0.55)]
        y_s = power_law(x_s, A1, alpha1)
        ax.annotate(
            "Exponent Change\n(lower exponent)",
            xy=(x_s, y_s),
            xytext=(x_s * 1.5, y_s * 1.8),
            fontsize=15,
            color=col_steeper,
            weight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.8, color=col_steeper),
        )

        x_sh = C[int(len(C) * 0.20)]
        y_sh = power_law(x_sh, A1 * shift_factor, alpha1)
        ax.annotate(
            "Shift (same exponent,\nlower constant)",
            xy=(x_sh, y_sh),
            xytext=(x_sh * 1.5, y_sh * 1.6),
            fontsize=15,
            color=col_shift,
            weight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.8, color=col_shift),
        )
        ax.annotate(
            "",
            xy=(x_sh, y_sh),
            xytext=(x_sh, power_law(x_sh, A1, alpha1)),
            arrowprops=dict(arrowstyle="<->", lw=1.6, color=col_shift, shrinkA=3, shrinkB=3),
        )
        ax.text(
            x_sh * 1.15,
            math.sqrt(y_sh * power_law(x_sh, A1, alpha1)),
            "shift",
            va="center",
            fontsize=10,
            color=col_shift,
        )

        ax.legend(loc="upper right", frameon=True, fontsize=11)
        fig.tight_layout()

    png_path = output_dir / "algorithmic_progress_types.png"
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    return close_and_return(fig, png_path)


def plot_ceg_stackplot(
    ax,
    graph_year_min=2017,
    graph_year_max=2025,
    C=None,
    innovations=None,
    label_offset=1.7,
    annotation_size=12,
    label_size=14.4,
    tick_size=14,
    show_ylabel=True,
):
    if innovations is None:
        innovations = INNOVATIONS

    if C is None:
        C_min = np.floor(ytc(graph_year_min))
        C_max = np.ceil(ytc(graph_year_max))
        C = np.array(sorted(list(np.logspace(np.log10(C_min), np.log10(C_max), 1000)) + [C_min, C_max]))
    C_label = ytc(graph_year_max)

    ceg_functions = compute_ceg_functions(C, innovations=innovations)
    cumulative_ceg_functions = compute_ceg_functions(C, innovations=innovations, cumulative=True)

    innovation_keys = list(innovations.keys())
    stackable_layers = [cumulative_ceg_functions[innovation_keys[0]]]
    for i in range(1, len(innovation_keys)):
        stackable_layers.append(cumulative_ceg_functions[innovation_keys[i]] - cumulative_ceg_functions[innovation_keys[i - 1]])

    ax.stackplot(C, np.vstack(stackable_layers), colors=[innovation["color"] for _, innovation in innovations.items()])

    previous_y_height = 1
    label_y_positions = stackable_layers[0][-1] ** (1 / 2)
    for innovation_key, innovation_config in innovations.items():
        color, label = innovation_config["color"], innovation_config["label"]
        cumulative_multiplier = cumulative_ceg_functions[innovation_key][-1]
        stack_height = cumulative_multiplier / previous_y_height
        midpoint = previous_y_height * (stack_height) ** (1 / 2)
        previous_y_height = cumulative_multiplier

        ax.plot([C_label, C_label * label_offset], [midpoint, label_y_positions], color=color, linewidth=1, alpha=0.7, clip_on=False)
        ax.text(
            C_label * label_offset,
            label_y_positions,
            f"{label} ({format_sigfigs(ceg_functions[innovation_key][-1])}x)",
            color=color,
            fontsize=annotation_size,
            va="center",
            fontweight="bold",
            clip_on=False,
        )
        label_y_positions *= 2.1

    total_gains = cumulative_ceg_functions["chinchilla"][-1]
    total_cagr = total_gains ** (1 / (graph_year_max - graph_year_min)) * 100 - 100
    ax.text(
        C_label * label_offset,
        6,
        f"Total Efficiency Gains:\n{format_sigfigs(total_gains)}x ({format_sigfigs(total_cagr)}% CAGR)",
        ha="left",
        va="top",
        fontsize=annotation_size * 1.1,
        color="black",
        fontweight="bold",
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ytc(graph_year_min), C_label)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel("Model Year", fontsize=label_size, fontstyle="italic")
    if show_ylabel:
        ax.set_ylabel("Compute Equivalent Multiplier\n(Relative to LSTM)", fontsize=label_size, fontstyle="italic")

    year_ticks = np.arange(np.ceil(graph_year_min), np.floor(graph_year_max))
    compute_values = [ytc(y) for y in year_ticks]
    ax.set_xticks(compute_values)
    ax.set_xticklabels([int(x) for x in year_ticks], fontsize=tick_size)
    ax.tick_params(labelsize=tick_size)
    ax.minorticks_off()

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ytc(graph_year_min), C_label)
    ax2.set_xlabel("Compute Frontier (FLOPs)", fontsize=label_size, fontstyle="italic", labelpad=10)
    ax2.tick_params(labelsize=tick_size)
    ax.set_title(
        "Scale Dependent Changes Dominate Efficiency Gains on Increasing Compute Frontier",
        fontweight="bold",
        fontsize=label_size * 1.1,
    )
    ax.text(ytc(2017.1), 3e4, "(a)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    return ax2


def save_scale_dependent_stackplot(output_dir: str | Path | None = None) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_ceg_stackplot(ax, 2016.9, 2025)
    fig.tight_layout()
    fig.set_dpi(600)
    png_path = output_dir / "scale_dependent_stackplot.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    return close_and_return(fig, png_path)


def plot_compute_equivalent_growth(
    ax,
    year_min,
    year_max,
    compute_value,
    label_size=12,
    tick_size=12,
    annotation_size=10,
    label_offset=0.35,
    show_ylabel=True,
):
    years = np.linspace(year_min, year_max, 1000)
    cumulative_ceg = compute_ceg_functions([compute_value] * 1000, years=years, cumulative=True)
    individual_ceg = compute_ceg_functions([compute_value] * 1000, years=years, cumulative=False)

    innovation_keys = list(INNOVATIONS.keys())
    stackable_layers = [cumulative_ceg[innovation_keys[0]]]
    for i in range(1, len(innovation_keys)):
        stackable_layers.append(cumulative_ceg[innovation_keys[i]] - cumulative_ceg[innovation_keys[i - 1]])

    ax.stackplot(years, np.vstack(stackable_layers), colors=[innovation["color"] for innovation in INNOVATIONS.values()])

    previous_y_height = 1
    label_y_positions = stackable_layers[0][-1] ** (1 / 2)
    year_label = year_max

    for innovation_key in innovation_keys:
        color = INNOVATIONS[innovation_key]["color"]
        label = INNOVATIONS[innovation_key]["label"]
        cumulative_multiplier = cumulative_ceg[innovation_key][-1]
        stack_height = cumulative_multiplier / previous_y_height
        midpoint = previous_y_height * (stack_height) ** (1 / 2)
        previous_y_height = cumulative_multiplier

        ax.plot([year_label, year_label + label_offset], [midpoint, label_y_positions], color=color, linewidth=1, alpha=0.7, clip_on=False)
        ax.text(
            year_label + label_offset,
            label_y_positions,
            f"{label} ({format_sigfigs(individual_ceg[innovation_key][-1])}x)",
            color=color,
            fontsize=annotation_size,
            va="center",
            fontweight="bold",
            clip_on=False,
        )
        label_y_positions *= 2.1

    final_value = cumulative_ceg["chinchilla"][-1]
    total_cagr = final_value ** (1 / (year_max - year_min)) * 100 - 100
    ax.text(
        year_label + label_offset,
        2,
        f"Total Efficiency Gains:\n{format_sigfigs(final_value)}x ({format_sigfigs(total_cagr)}% CAGR)",
        ha="left",
        va="top",
        fontsize=annotation_size * 1.1,
        color="black",
        fontweight="bold",
        clip_on=False,
    )

    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    if show_ylabel:
        ax.set_ylabel("Compute Equivalent Multiplier\n(Relative to LSTM)", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xlim(year_min, year_max)
    ax.minorticks_off()

    exponent = round(np.log10(compute_value))
    if not np.isclose(exponent, np.log10(compute_value)):
        raise ValueError("compute_value must be an exact power of 10")
    ax.set_title(rf"Decomposed Efficiency Gains at $10^{{{exponent}}}$ FLOPs, Over Time", fontsize=label_size * 1.2, fontweight="bold")
    ax.text(2017.1, 3e4, "(b)", fontweight="bold")
    return ax


def plot_compute_silhouette(
    ax,
    year_min,
    year_max,
    compute_values,
    ylabel_fontsize=16,
    tick_fontsize=12,
    annotation_size=12,
    label_offset=0.35,
    show_ylabel=True,
    show_cagr=True,
    linewidth=2.5,
    labels_inside=False,
    show_gridlines=True,
    subplot_label=True,
):
    granularity = 1000
    years = np.linspace(year_min, year_max, granularity)
    compute_values = list(map(np.float64, compute_values))
    cmap = plt.cm.viridis_r
    color_offset = 2
    norm = Normalize(vmin=0, vmax=len(compute_values) - 1 + color_offset)
    line_endpoints = []

    for i, c in enumerate(compute_values):
        cumulative_ceg = compute_ceg_functions([c] * granularity, years=years, cumulative=True)
        silhouette = cumulative_ceg["chinchilla"]
        color = cmap(norm(i + color_offset))
        ax.plot(years, silhouette, color=color, linewidth=linewidth)
        cagr = ((silhouette[-1] / silhouette[0]) ** (1 / (year_max - year_min)) - 1) * 100
        line_endpoints.append((silhouette[-1], color, int(np.log10(float(c))), cagr))

    ax.set_yscale("log")
    ax.set_xlim(year_min, year_max)
    ax.set_ylim(1, 1e5)
    if show_ylabel:
        ax.set_ylabel("Compute Equivalent Multiplier\n(Relative to LSTM)", fontsize=ylabel_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if show_gridlines:
        ax.grid(True, which="major", alpha=0.3)
    else:
        ax.grid(False)

    year_range = year_max - year_min
    if year_range <= 10:
        ax.set_xticks(range(int(np.ceil(year_min)), int(year_max) + 1))
    else:
        ax.set_xticks(range(int(np.ceil(year_min)), int(year_max) + 1, 2))

    label_x = 2022.5 if labels_inside else year_max + label_offset
    label_ha = "left"
    for y_val, color, log_c, cagr in line_endpoints:
        if show_cagr:
            label_text = f"$\\mathbf{{10^{{{log_c}}}}}$ FLOPs ({cagr:.0f}% CAGR)"
        else:
            label_text = f"$\\mathbf{{10^{{{log_c}}}}}$ FLOPs"
        ax.text(label_x, y_val * (10**0.15), label_text, color=color, fontsize=annotation_size, va="center", ha=label_ha)

    ax.set_title("Efficiency Gains Over Time With Fixed Compute", fontsize=ylabel_fontsize * 1.2, fontweight="bold")
    if subplot_label:
        ax.text(2017.1, 3e4, "(c)", fontweight="bold")
    return ax


def save_combined_scale_dependence_graphs(output_dir: str | Path | None = None) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    label_size, tick_size, annotation_size = 15, 13, 11
    graph_year_min, graph_year_max = 2016.5, 2023

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 10))
    plot_ceg_stackplot(
        axs[0],
        graph_year_min,
        graph_year_max,
        label_size=label_size,
        tick_size=tick_size,
        annotation_size=annotation_size,
        show_ylabel=False,
    )
    plot_compute_equivalent_growth(
        axs[1],
        graph_year_min,
        graph_year_max,
        1e18,
        label_size=label_size,
        tick_size=tick_size,
        annotation_size=annotation_size,
        show_ylabel=False,
    )
    plot_compute_silhouette(
        axs[2],
        graph_year_min,
        graph_year_max,
        [10**i for i in range(16, 25, 2)],
        ylabel_fontsize=label_size,
        tick_fontsize=tick_size,
        annotation_size=annotation_size + 2,
        show_ylabel=False,
    )

    fig.supylabel("Compute Equivalent Multiplier\n(Relative to LSTM)", fontsize=label_size * 1.2, x=0.02, ha="center")
    fig.tight_layout()
    fig.set_dpi(600)

    png_path = output_dir / "combined_scale_dependence_graphs.png"
    fig.savefig(png_path, dpi=600)
    return close_and_return(fig, png_path)


_cmap = cm.viridis_r
KAPLAN_COLOR = _cmap(0.5)
DENSE_COLOR = _cmap(0.7)
LSTM_COLOR = _cmap(0.9)


def plot_ceg_growth_rates(
    axes,
    C_MIN=1e19,
    C_MAX=1e26,
    label_fontsize=12,
    tick_fontsize=10,
    include_title=True,
    year_skip=2,
    **_unused,
):
    if len(axes) != 3:
        raise ValueError("Must provide exactly 3 axes")

    C = np.logspace(np.log10(C_MIN) - 1, np.log10(C_MAX) + 1, 1000)
    C = C[(C >= C_MIN) & (C <= C_MAX)]
    years = np.array([cty(c) for c in C])

    series = [
        np.gradient(np.log(mixture_of_experts_multiplier(C) * lstm_to_optimal_transformer_multiplier(C, kaplan_divide=False)), years) * 100,
        np.gradient(np.log(mixture_of_experts_multiplier(C) * chinchilla_rebalancing_multiplier(C)), years) * 100,
        np.gradient(np.log(mixture_of_experts_multiplier(C)), years) * 100,
    ]
    colors = [LSTM_COLOR, KAPLAN_COLOR, DENSE_COLOR]
    titles = [
        "Measured Growth Rate Relative to LSTM",
        "Measured Growth Rate Relative to Kaplan Transformer",
        "Measured Growth Relative to Dense Transformer",
    ]

    min_year = int(np.floor(cty(C_MIN)))
    max_year = int(np.ceil(cty(C_MAX)))
    year_range = np.arange(min_year, max_year + 1)
    threshold = 0.1

    for i, (ax, data, color, title) in enumerate(zip(axes, series, colors, titles)):
        bars = []
        for year in year_range:
            compute = ytc(year)
            idx = np.argmin(np.abs(C - compute))
            bars.append(data[idx])

        years_ticks = np.arange(2018, max_year + 1, year_skip)
        ax.set_xticks(years_ticks)
        ax.set_xticklabels([int(year) for year in years_ticks])
        ax.minorticks_off()

        for year, value in zip(year_range, bars):
            if value < threshold:
                ax.bar(year, threshold, color=color, edgecolor=color, linewidth=3, alpha=0.8)
            else:
                ax.bar(year, value, color=color, alpha=0.8)

        if include_title:
            ax.set_title(title, fontsize=label_fontsize)
        ax.set_ylim(-2, 100)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(True, alpha=0.5, axis="y")
        if i == 0:
            ax.set_ylabel("Annual CEG\nGrowth Rate", fontsize=label_fontsize, fontstyle="italic")
        ax.set_xlim(min_year - 0.5, max_year + 0.5)
        if i == 1:
            ax.set_xlabel("Model Year", fontsize=label_fontsize, fontstyle="italic")
        ax.tick_params(axis="x", which="both", labelsize=tick_fontsize)
        ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)


def plot_ceg_multipliers(
    axes,
    normalize=False,
    C_MIN=None,
    C_MAX=None,
    show_top_xaxis=True,
    show_bottom_xaxis=True,
    show_top_xlabel=True,
    show_bottom_xlabel=True,
    label_fontsize=12,
    multiplier_font_size=30,
    tick_fontsize=20,
    year_skip=2,
):
    if len(axes) != 3:
        raise ValueError("axes must contain exactly 3 axes")

    if C_MIN is None:
        C_MIN = ytc(2018)
    if C_MAX is None:
        C_MAX = 1e30

    C = np.logspace(np.log10(C_MIN), np.log10(C_MAX), 1000)
    C = C[(C >= C_MIN) & (C <= C_MAX)]
    line_configs = [
        ("lstm", mixture_of_experts_multiplier(C) * lstm_to_optimal_transformer_multiplier(C, kaplan_divide=False), LSTM_COLOR, "Relative to\nLSTM", None),
        ("kaplan", mixture_of_experts_multiplier(C) * chinchilla_rebalancing_multiplier(C), KAPLAN_COLOR, "Relative to\nKaplan Transformer", ytc(2022)),
        ("dense", mixture_of_experts_multiplier(C), DENSE_COLOR, "Relative to\nDense Transformer", ytc(2021)),
    ]

    for idx, (line_type, values, color, title, threshold) in enumerate(line_configs):
        ax = axes[idx]
        if normalize:
            if line_type == "lstm":
                values = values / lstm_to_optimal_transformer_multiplier(np.array([C_MIN]), kaplan_divide=False)[0]
            elif line_type == "kaplan":
                values = values / chinchilla_rebalancing_multiplier(np.array([C_MIN]))[0]
            elif line_type == "dense":
                values = values / mixture_of_experts_multiplier(np.array([C_MIN]))[0]

        if threshold:
            mask_before = C < threshold
            mask_after = C >= threshold
            ax.plot(C[mask_before], values[mask_before], color=color, linewidth=3, linestyle="--", alpha=0.8)
            ax.plot(C[mask_after], values[mask_after], color=color, linewidth=3, alpha=0.8)
            ax.fill_between(C[mask_after], values[mask_after], alpha=0.8, color=color)
        else:
            ax.plot(C, values, color=color, linewidth=3, alpha=0.8)
            ax.fill_between(C, values, alpha=0.8, color=color)

        final_value = values[-1]
        ax.text(
            C[-1],
            5,
            f"{format_sigfigs(final_value, sigfigs=2)}x",
            fontsize=multiplier_font_size,
            fontweight="bold",
            color="white",
            ha="right",
            va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(C_MIN, C_MAX)
        max_year = np.ceil(cty(C_MAX)) - 1
        years_ticks = np.arange(2018, max_year + 1, year_skip)
        tick_locs = [ytc(year) for year in years_ticks]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([int(year) for year in years_ticks])
        ax.minorticks_off()
        ax.set_yticks([10**i for i in range(1, 6) if i % 2 == 1])

        if show_bottom_xlabel and idx == 1:
            ax.set_xlabel("Model Year", fontsize=label_fontsize, fontstyle="italic")
        else:
            ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=show_bottom_xaxis, labelsize=tick_fontsize)

        if idx == 0:
            if normalize:
                year = round(cty(C_MIN))
                ylabel = f"Compute Equivalent Gains\nIncrease Since {year}"
            else:
                ylabel = "CEG Multiplier"
            ax.set_ylabel(ylabel, fontsize=label_fontsize, fontstyle="italic", labelpad=60)

        ax.grid(True, alpha=0.5)
        ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)
        ax.set_title(title, fontsize=label_fontsize, y=1.4, fontweight="bold")

        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(C_MIN, C_MAX)
        if show_top_xlabel and idx == 1:
            ax2.set_xlabel("Compute Frontier (FLOPs)", fontsize=label_fontsize, fontstyle="italic")
            ax2.xaxis.set_label_coords(0.5, 1.25)
        else:
            ax2.set_xlabel("")
        ax2.tick_params(axis="x", which="both", labeltop=show_top_xaxis, labelsize=tick_fontsize)


def save_ceg_reference_combined(output_dir: str | Path | None = None) -> list[Path]:
    output_dir = ensure_output_dir(output_dir)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharey="row")
    C_MIN, C_MAX = ytc(2017), ytc(2028)
    label_size, title_size, tick_size, year_skip = 21, 23, 24, 4

    plot_ceg_multipliers(
        axs[0],
        C_MIN=C_MIN,
        C_MAX=C_MAX,
        show_bottom_xaxis=True,
        label_fontsize=label_size,
        tick_fontsize=tick_size,
        normalize=False,
        show_bottom_xlabel=False,
        year_skip=year_skip,
    )
    plot_ceg_growth_rates(
        axs[1],
        C_MIN=C_MIN,
        C_MAX=C_MAX,
        show_top_xaxis=False,
        label_fontsize=label_size,
        tick_fontsize=tick_size,
        yaxis_right=False,
        include_title=False,
        year_skip=year_skip,
    )

    fig.suptitle("Growth in Compute Equivalent Gains Depends on the Reference Algorithm", fontsize=title_size, y=0.987, fontweight="bold")
    fig.set_dpi(300)
    fig.tight_layout(h_pad=0.5)
    png_path = output_dir / "CEG_reference_combined.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    return close_and_return(fig, png_path)
