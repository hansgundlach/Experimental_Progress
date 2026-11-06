# %%
"""
NeurIPS-Style Two-Panel Figure — LR-Trend Datasets

Same layout as plot_analysis_final.py but using:
  - x1_modern_all_lr_trend   (modern transformer, trend LR)
  - x1_lstm_layer1_lr_trend  (LSTM layer 1, trend LR)
  - x1_all_lr_historical_trend (historical transformer, trend LR)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats as _scipy_stats

from graphing_utils import TrainingCurveAnalyzer, ALPHA_CONFIG

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "new_experiments_folder_1"
IRREDUCIBLE_LOSS = 1.9

NEURIPS_FONT_CONFIG = {
    "xlabel_size": 14,
    "ylabel_size": 16,
    "title_size": 16,
    "title_weight": "bold",
    "major_tick_size": 16,
    "minor_tick_size": 16,
    "legend_size": 13,
    "equation_size": 17,
    "annotation_size": 15,
    "target_loss_label_size": 14,
}

USE_THEORETICAL_FLOPS = False

class_legend_mapping = {
    "modern_transformer": "Modern Transformer",
    "lstm_layer1":        "LSTM (1 layer)",
    "historical":         "Historical Transformer",
}

# ── Experiment dims ────────────────────────────────────────────────────────────
MODERN_DIMS   = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
HIST_DIMS     = [32, 48, 64, 80, 96, 112, 160, 192, 224, 256]   # 128d diverged
LSTM1_DIMS    = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 352, 384]
                 

STOP, END = 1e16, 1e17

# ── Build analyzer ─────────────────────────────────────────────────────────────
analyzer = TrainingCurveAnalyzer(
    irreducible_loss=IRREDUCIBLE_LOSS,
    use_theoretical_flops=USE_THEORETICAL_FLOPS,
    class_legend_mapping=class_legend_mapping,
)

def add_dim_sweep(folder, class_name, dims, color, label_prefix=""):
    for dim in dims:
        path = DATA_ROOT / folder / f"{dim}d.csv"
        if not path.exists():
            print(f"  SKIP (missing): {folder}/{dim}d.csv")
            continue
        analyzer.add_experiment(
            name=f"{dim}d {class_name}",
            csv_path=str(path),
            color=color,
            marker="o",
            include_in_frontier=True,
            class_name=class_name,
            hidden_dim=dim,
        )

add_dim_sweep("x1_modern_all_lr_trend",   "modern_transformer", MODERN_DIMS, "viridis[0.0]")
add_dim_sweep("x1_lstm_layer1_lr_trend",  "lstm_layer1",        LSTM1_DIMS,  "viridis[0.8]")
add_dim_sweep("x1_all_lr_historical_trend","historical",         HIST_DIMS,   "viridis[0.6]")


# ── Panel helper (mirrors plot_analysis_final.py) ──────────────────────────────
def plot_panel_to_axis(
    ax, classes_to_plot, flop_range_by_class, extrapolation_range,
    show_power_law_fit=True, theoretical_scaling_laws=None,
    panel_label=None, target_loss_lines=None,
    equation_positions=None, show_ylabel=True, title=None,
    show_legend=False,
):
    # Include any extra classes needed for target loss line comparisons
    extra_classes = []
    if target_loss_lines:
        for line_cfg in target_loss_lines:
            for cls in line_cfg.get("classes", []):
                if cls not in classes_to_plot and cls not in extra_classes:
                    extra_classes.append(cls)
    all_frontier_classes = classes_to_plot + extra_classes
    full_flop_range = dict(flop_range_by_class)
    for cls in extra_classes:
        if cls not in full_flop_range:
            full_flop_range[cls] = (STOP, END)
    analyzer.identify_frontier_by_class(
        method="pareto", flop_range=None, use_all_points=True,
        classes=all_frontier_classes, flop_range_by_class=full_flop_range,
    )

    plotted_classes = set()
    for name, exp in analyzer.experiments.items():
        cls = exp.get("class", "default")
        if cls not in classes_to_plot:
            continue
        color = exp.get("color", "tab:blue")
        compute_vals = exp["data"][exp["compute_col"]].values
        loss_vals = exp["data"][exp["loss_col"]].values - analyzer.irreducible_loss
        mask = loss_vals > 0
        if not np.any(mask):
            continue
        legend_label = analyzer.class_legend_mapping.get(cls, cls)
        ax.plot(
            compute_vals[mask], loss_vals[mask],
            marker=exp["marker"], linestyle=exp["linestyle"],
            color=color, alpha=ALPHA_CONFIG["data_points_alpha"],
            label=legend_label if cls not in plotted_classes else None,
            linewidth=1.5, markersize=3,
        )
        plotted_classes.add(cls)

    for cls in classes_to_plot:
        pts = analyzer.frontier_points_all_by_class.get(cls, [])
        for exp_name, comp, loss in pts:
            color = analyzer.experiments[exp_name].get("color", "k") if exp_name in analyzer.experiments else "k"
            ax.scatter(comp, loss, color=color, s=120, marker="*", zorder=120,
                       edgecolors="black", linewidth=1.5,
                       alpha=ALPHA_CONFIG["frontier_points_alpha"], label=None)

    if show_power_law_fit:
        fit_results = analyzer.fit_power_law_by_class(class_names=classes_to_plot, use_all_points=True)
        x_min, x_max = extrapolation_range
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        for cls in classes_to_plot:
            params = fit_results.get(cls)
            if params is None:
                continue
            a, b, r2 = params
            print(f"  {class_legend_mapping.get(cls, cls)}: L-E = {a:.3f}·C^({b:.4f}), R²={r2:.4f}")
            class_color = analyzer.get_class_color(cls)
            fit_label = None if not show_legend else f"{class_legend_mapping.get(cls, cls)} fit"
            ax.plot(x_fit, a * np.power(x_fit, b), "--", linewidth=3,
                    alpha=ALPHA_CONFIG["power_law_fit_alpha"], color=class_color,
                    label=fit_label)
            if equation_positions is None:
                # auto-position
                mid = len(x_fit) // 2
                eq_x, eq_y = x_fit[mid], (a * np.power(x_fit[mid], b)) * 1.3
                ax.text(eq_x, eq_y, f"${a:.1f} \\cdot C^{{{b:.3f}}}$",
                        fontsize=NEURIPS_FONT_CONFIG["equation_size"], color="black",
                        weight="bold", ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                  alpha=0.8, edgecolor=class_color, linewidth=2))
            elif cls in equation_positions:
                eq_x, eq_y = equation_positions[cls]
                ax.text(eq_x, eq_y, f"${a:.1f} \\cdot C^{{{b:.3f}}}$",
                        fontsize=NEURIPS_FONT_CONFIG["equation_size"], color="black",
                        weight="bold", ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                  alpha=0.8, edgecolor=class_color, linewidth=2))
            # else: equation_positions={} → suppress box entirely

    if theoretical_scaling_laws:
        x_min, x_max = extrapolation_range
        x_th = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        for law in theoretical_scaling_laws:
            E, A, gamma = law.get("E", IRREDUCIBLE_LOSS), law.get("A", 1.0), law.get("gamma", -0.1)
            y_th = (E - analyzer.irreducible_loss) + A * np.power(x_th, gamma)
            th_label = f"{law.get('label','Theory')}:\n{A:.1f}·C^({gamma:.3f})" if show_legend else None
            ax.plot(x_th, y_th, color=law.get("color","red"),
                    linestyle=law.get("linestyle","--"), linewidth=law.get("linewidth",3),
                    alpha=law.get("alpha",0.8), label=th_label)

    if target_loss_lines and show_power_law_fit:
        # Fit all classes referenced in compare_classes, not just plotted ones
        all_compare_classes = list({
            cls for line_cfg in target_loss_lines
            for cls in line_cfg.get("classes", [])
        })
        fit_results = analyzer.fit_power_law_by_class(
            class_names=list(set(classes_to_plot + all_compare_classes)), use_all_points=True
        )
        for line_cfg in target_loss_lines:
            target_loss = line_cfg.get("loss")
            compare_classes = line_cfg.get("classes", [])
            if len(compare_classes) != 2:
                continue
            intersections = {}
            for cls in compare_classes:
                if cls in fit_results and fit_results[cls] is not None:
                    a, b, _ = fit_results[cls]
                    if b != 0 and a > 0 and target_loss > 0:
                        intersections[cls] = np.power(target_loss / a, 1.0 / b)
            if len(intersections) != 2:
                continue
            c1, c2 = compare_classes
            v1, v2 = intersections.get(c1), intersections.get(c2)
            if v1 is None or v2 is None:
                continue
            if v1 > v2:
                v1, v2, c1, c2 = v2, v1, c2, c1
            ratio = float(v2) / float(v1)
            ax.axhline(y=target_loss, color="black", linestyle="--",
                       linewidth=2.5, alpha=0.8, zorder=85)
            ax.text(0.02, target_loss * 1.05, f"{target_loss:.1f}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=NEURIPS_FONT_CONFIG["target_loss_label_size"],
                    color="black", ha="left", va="bottom", zorder=95)
            ax.plot([float(v1), float(v2)], [target_loss, target_loss],
                    color=line_cfg.get("color","orange"), linestyle="-",
                    linewidth=5, alpha=1.0, zorder=90)
            mid = np.sqrt(float(v1) * float(v2))
            y_ann = float(target_loss) * line_cfg.get("annotation_y_multiplier", 1.3)
            ax.annotate(f"{ratio:.1f}x", xy=(mid, target_loss), xytext=(mid, y_ann),
                        fontsize=NEURIPS_FONT_CONFIG["annotation_size"], color="black",
                        weight="bold", ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="->", color="black", lw=2, alpha=0.7),
                        zorder=100)
            print(f"  Target {target_loss:.2f}: {c1}={v1:.2e}, {c2}={v2:.2e}, ratio={ratio:.2f}x")

    ax.set_xlabel("Compute (FLOPs)", fontsize=NEURIPS_FONT_CONFIG["xlabel_size"])
    if show_ylabel:
        ax.set_ylabel("Validation Loss - Irreducible", fontsize=NEURIPS_FONT_CONFIG["ylabel_size"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    fmt = ticker.ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=NEURIPS_FONT_CONFIG["major_tick_size"])
    ax.tick_params(axis="both", which="minor", labelsize=NEURIPS_FONT_CONFIG["minor_tick_size"])
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=NEURIPS_FONT_CONFIG["title_size"], fontweight="bold",
                va="top", ha="left")
    if title:
        ax.set_title(title, fontsize=NEURIPS_FONT_CONFIG["title_size"],
                     fontweight=NEURIPS_FONT_CONFIG["title_weight"])
    if show_legend:
        leg = ax.legend(loc="upper right", fontsize=NEURIPS_FONT_CONFIG["legend_size"],
                        framealpha=0.9, prop={"weight": "bold"})
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", None)
        if handles:
            for h in handles:
                if hasattr(h, "set_alpha"): h.set_alpha(1.0)


# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)


# Panel 1: Modern Transformer vs LSTM layer 1
ax1 = fig.add_subplot(gs[0, 0])
print("Panel 1 fits:")
plot_panel_to_axis(
    ax=ax1,
    classes_to_plot=["modern_transformer", "lstm_layer1"],
    flop_range_by_class={
        "modern_transformer": (STOP, END),
        "lstm_layer1":        (STOP, END),
    },
    extrapolation_range=(1e14, 1e18),
    show_power_law_fit=True,
    panel_label="(a)",
    target_loss_lines=[
        {"loss": 3.377, "classes": ["modern_transformer", "lstm_layer1"],
         "color": "orange", "linewidth": 4, "annotation_y_multiplier": 1.3},
        {"loss": 2.5,   "classes": ["modern_transformer", "lstm_layer1"],
         "color": "orange", "linewidth": 4, "annotation_y_multiplier": 1.2},
    ],
    equation_positions={
        "modern_transformer": (2e14, 2.0),
        "lstm_layer1":        (5e16, 5.7),
    },
    show_ylabel=True,
    title="Modern Transformer vs LSTM Scaling",
)

# Pre-compute modern transformer fit to use as reference in panel 2
analyzer.identify_frontier_by_class(
    method="pareto", flop_range=None, use_all_points=True,
    classes=["modern_transformer"],
    flop_range_by_class={"modern_transformer": (STOP, END)},
)
_mod_fit = analyzer.fit_power_law_by_class(class_names=["modern_transformer"], use_all_points=True)
_mod_A, _mod_gamma, _mod_r2 = _mod_fit.get("modern_transformer", (74.1, -0.090, 0.0))

# Panel 2: Historical transformer vs modern transformer (theoretical)
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
print("\nPanel 2 fits:")
plot_panel_to_axis(
    ax=ax2,
    classes_to_plot=["historical"],
    flop_range_by_class={
        "historical": (STOP, END),
    },
    extrapolation_range=(1e14, 1e18),
    show_power_law_fit=True,
    theoretical_scaling_laws=[
        {
            "E": 1.9, "A": _mod_A, "gamma": _mod_gamma,
            "label": "Modern Transformer",
            "color": "purple", "linestyle": "--", "linewidth": 3, "alpha": 0.8,
        },
    ],
    panel_label="(b)",
    target_loss_lines=[
        {"loss": 3.377, "classes": ["historical", "modern_transformer"],
         "color": "orange", "linewidth": 4, "annotation_y_multiplier": 1.3},
        {"loss": 2.5,   "classes": ["historical", "modern_transformer"],
         "color": "orange", "linewidth": 4, "annotation_y_multiplier": 1.2},
    ],
    equation_positions={
        "historical": (7e16, 5.7),
    },
    show_ylabel=False,
    title="Historical vs Modern Transformer Scaling",
)
plt.setp(ax2.get_yticklabels(), visible=False)

plt.tight_layout()

save_path = ROOT / "Figures" / "neurips_two_panel_lr_trend.png"
save_path.parent.mkdir(exist_ok=True)
fig.savefig(save_path, dpi=300, bbox_inches="tight")
fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
print(f"\nSaved: {save_path}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SCALING LAW FIT TABLE — LaTeX output
# Recomputes fits with scipy.stats.linregress on the Pareto frontier points
# to obtain standard errors and 95% CIs on the exponent alpha.
# Saved to Figures/scaling_law_fits_table.tex each run.
# ══════════════════════════════════════════════════════════════════════════════
def generate_fit_table():
    """
    Fit L-E = A * C^alpha via log-log OLS on the Pareto frontier points,
    returning alpha, 95% CI on alpha, A, and R² for each class.
    Uses the same frontier already identified for the two-panel figure
    (STOP, END flop range, IRREDUCIBLE_LOSS = 1.9).
    """
    # Re-identify frontier with all three classes together at (STOP, END)
    table_classes = ["modern_transformer", "lstm_layer1", "historical"]
    analyzer.identify_frontier_by_class(
        method="pareto", flop_range=None, use_all_points=True,
        classes=table_classes,
        flop_range_by_class={c: (STOP, END) for c in table_classes},
    )

    display_names = {
        "modern_transformer": "Modern Transformer",
        "lstm_layer1":        "LSTM (1 layer)",
        "historical":         "Historical Transformer",
    }

    rows = []
    for cls in table_classes:
        pts = analyzer.frontier_points_all_by_class.get(cls, [])
        if len(pts) < 3:
            print(f"  WARNING: too few frontier points for {cls} ({len(pts)}), skipping table row")
            continue
        log_C = np.log(np.array([p[1] for p in pts]))
        log_L = np.log(np.array([p[2] for p in pts]))

        res = _scipy_stats.linregress(log_C, log_L)
        alpha   = res.slope
        A       = np.exp(res.intercept)
        r2      = res.rvalue ** 2
        se      = res.stderr          # SE on slope (alpha)
        n       = len(pts)
        t_crit  = _scipy_stats.t.ppf(0.975, df=n - 2)
        ci      = t_crit * se         # half-width of 95% CI

        rows.append({
            "cls": cls,
            "name": display_names[cls],
            "alpha": alpha,
            "ci": ci,
            "A": A,
            "r2": r2,
            "n": n,
        })
        print(f"  {display_names[cls]}: alpha={alpha:.4f} ± {ci:.4f} (95% CI, n={n}), A={A:.2f}, R²={r2:.4f}")

    # ── Build LaTeX table ───────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Exponent $\alpha$ (95\% CI) & Coefficient $A$ & $R^2$ \\")
    lines.append(r"\midrule")
    for r in rows:
        alpha_str = f"${r['alpha']:.3f} \\pm {r['ci']:.3f}$"
        A_str     = f"${r['A']:.1f}$"
        r2_str    = f"${r['r2']:.3f}$"
        lines.append(f"{r['name']} & {alpha_str} & {A_str} & {r2_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Power-law scaling fits $L - \hat{L} = A \cdot C^{\alpha}$ for each architecture, "
        r"where $\hat{L} = " + f"{IRREDUCIBLE_LOSS}" + r"$ is the assumed irreducible loss and $C$ is "
        r"total training compute (FLOPs). "
        r"Fits use the Pareto frontier of training curves over "
        r"$C \in [10^{" + f"{int(np.log10(STOP))}" + r"}, 10^{" + f"{int(np.log10(END))}" + r"}]$ FLOPs. "
        r"Confidence intervals are 95\% from OLS on the log-log frontier.}"
    )
    lines.append(r"\label{tab:scaling_fits}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    tex_path = ROOT / "Figures" / "scaling_law_fits_table.tex"
    tex_path.parent.mkdir(exist_ok=True)
    tex_path.write_text(tex)
    print(f"\nLaTeX table saved: {tex_path}")
    print("\n" + tex)
    return tex

print("\nScaling law fit table:")
generate_fit_table()


# %%
# ══════════════════════════════════════════════════════════════════════════════
# IRREDUCIBLE LOSS SENSITIVITY ANALYSIS — PARAMETERS
# Modify the values below to control the sweep.
# ══════════════════════════════════════════════════════════════════════════════

# Range of assumed irreducible loss values to sweep over
SENS_L0_VALUES = [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]

# Target loss levels (L - L0) at which compute ratios are computed.
# These match the orange lines in the two-panel figure.
SENS_TARGET_LOSSES = (3.377, 2.5)

# FLOP range used for frontier identification and fitting (same as main figure).
# Change these if you want a different fitting range for the sensitivity analysis.
SENS_FLOP_RANGE = (STOP, END)   # currently (1e16, 1e17); set by STOP, END above

# Output filenames (relative to project root / Figures/).
SENS_FIGURE_PATH = "Figures/E_sensitivity.png"
SENS_TABLE_PATH  = "Figures/E_sensitivity_table.csv"

# Classes and their data sources.
# Each entry: class_name -> (folder_under_new_experiments_folder_1, dims_list, color_spec)
# Add or remove entries here to include/exclude architectures.
SENS_CLASSES = {
    "modern_transformer": (
        "x1_modern_all_lr_trend",     # folder containing {dim}d.csv files
        MODERN_DIMS,                   # dims defined at top of file
        "viridis[0.0]",                # color (matches main figure)
    ),
    "lstm_layer1": (
        "x1_lstm_layer1_lr_trend",
        LSTM1_DIMS,
        "viridis[0.8]",
    ),
    "historical": (
        "x1_all_lr_historical_trend",
        HIST_DIMS,
        "viridis[0.6]",
    ),
}

# Compute ratio comparisons: list of (reference_class, comparison_class) pairs.
# The ratio reported is: compute(comparison) / compute(reference) at each target loss.
SENS_RATIO_PAIRS = [
    ("modern_transformer", "lstm_layer1"),
    ("modern_transformer", "historical"),
]

# ── Irreducible loss sensitivity function ──────────────────────────────────────
def run_sensitivity():
    """
    Sweep IRREDUCIBLE_LOSS over SENS_L0_VALUES and show how fitted exponents
    and compute ratios change. Reads from the folders defined in SENS_CLASSES.
    Produces:
      - SENS_TABLE_PATH  : CSV table of alpha, A, R² and compute ratios per L0
      - SENS_FIGURE_PATH : plot of |alpha| vs L0 for each architecture
    """
    classes = list(SENS_CLASSES.keys())

    def parse_color(spec):
        if "[" in spec and spec.endswith("]"):
            cm, idx = spec[:-1].split("[", 1)
            return plt.get_cmap(cm)(float(idx))
        return spec

    rows = []
    alpha_by_class = {c: [] for c in classes}

    for L0 in SENS_L0_VALUES:
        an = TrainingCurveAnalyzer(
            irreducible_loss=L0,
            use_theoretical_flops=False,
            class_legend_mapping=class_legend_mapping,
        )
        for cls, (folder, dims, color) in SENS_CLASSES.items():
            for dim in dims:
                path = DATA_ROOT / folder / f"{dim}d.csv"
                if path.exists():
                    an.add_experiment(
                        name=f"{dim}d {cls}", csv_path=str(path),
                        color=color, marker="o",
                        include_in_frontier=True, class_name=cls, hidden_dim=dim,
                    )

        flop_range_by_class = {c: SENS_FLOP_RANGE for c in classes}
        an.identify_frontier_by_class(
            method="pareto", flop_range=None, use_all_points=True,
            classes=classes, flop_range_by_class=flop_range_by_class,
        )
        fits = an.fit_power_law_by_class(class_names=classes, use_all_points=True)

        row = {"L0": L0}
        for cls in classes:
            p = fits.get(cls)
            if p:
                A, alpha, r2 = p
                row[f"{cls}_alpha"] = alpha
                row[f"{cls}_A"]     = A
                row[f"{cls}_r2"]    = r2
                alpha_by_class[cls].append(alpha)
            else:
                row[f"{cls}_alpha"] = np.nan
                alpha_by_class[cls].append(np.nan)

        for tl in SENS_TARGET_LOSSES:
            for ref, comp in SENS_RATIO_PAIRS:
                ref_p  = fits.get(ref)
                comp_p = fits.get(comp)
                if ref_p and comp_p:
                    A_r, b_r, _ = ref_p
                    A_c, b_c, _ = comp_p
                    try:
                        c_ref  = np.power(tl / A_r, 1.0 / b_r)
                        c_comp = np.power(tl / A_c, 1.0 / b_c)
                        ratio  = c_comp / c_ref if c_ref > 0 else np.nan
                    except Exception:
                        ratio = np.nan
                    row[f"ratio_{comp}_vs_{ref}_at_{tl:.2f}"] = ratio
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = ROOT / SENS_TABLE_PATH
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"\nSensitivity table saved: {csv_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Figure: |alpha| vs L_0 for each class ──────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
    for cls in classes:
        _, _, color_spec = SENS_CLASSES[cls]
        ax_s.plot(SENS_L0_VALUES, [-a for a in alpha_by_class[cls]],
                  "o-", linewidth=2, markersize=6,
                  color=parse_color(color_spec),
                  label=class_legend_mapping.get(cls, cls))

    ax_s.axvline(IRREDUCIBLE_LOSS, color="black", linestyle="--", linewidth=1.5,
                 alpha=0.7, label=f"Default L₀ = {IRREDUCIBLE_LOSS}")
    ax_s.set_xlabel("Assumed irreducible loss L₀", fontsize=13)
    ax_s.set_ylabel("Fitted scaling exponent |α|", fontsize=13)
    ax_s.set_title("Sensitivity to irreducible loss assumption", fontsize=14, fontweight="bold")
    ax_s.legend(fontsize=10, framealpha=0.9)
    ax_s.grid(True, alpha=0.3)
    ax_s.tick_params(labelsize=11)
    fig_s.tight_layout()

    png_path = ROOT / SENS_FIGURE_PATH
    fig_s.savefig(png_path, dpi=300, bbox_inches="tight")
    fig_s.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_s)
    print(f"Sensitivity figure saved: {png_path}")
    return df


if __name__ == "__main__" or True:
    sensitivity_df = run_sensitivity()
# %%
