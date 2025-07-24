# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# %%
# Replace all individual CSV loading with a unified experiment manager
class ExperimentManager:
    def __init__(self, base_path="../experimental_data_folder/"):
        self.base_path = base_path
        self.experiments = {}

    def load_experiment_group(self, group_name, configs):
        """Load a group of related experiments"""
        print(f"Loading {group_name}...")
        group_data = {}
        for config_name, file_path in configs.items():
            try:
                df = pd.read_csv(self.base_path + file_path)
                group_data[config_name] = {"data": df, "final_metrics": df.iloc[-1]}
                print(f"  ✓ {config_name}: {len(df)} steps")
            except FileNotFoundError:
                print(f"  ✗ {config_name}: File not found - {file_path}")
                continue
        self.experiments[group_name] = group_data
        return group_data


# Replace hardcoded experiment definitions with configuration
EXPERIMENT_CONFIGS = {
    "hidden_dim_scaling": {
        "16d": "Hidden_Dim_Scaling/16d.csv",
        "24d": "Hidden_Dim_Scaling/24d_123.csv",
        "32d": "Hidden_Dim_Scaling/32d_123.csv",
        "64d": "Hidden_Dim_Scaling/64d_123.csv",
        "96d": "Hidden_Dim_Scaling/96d_123.csv",
        "128d": "Hidden_Dim_Scaling/128d.csv",
    },
    "hidden_dim_no_rotary_123": {
        "16d": "Hidden_Dim_Scaling_No_Rotary_123/16d_no_rotary_123.csv",
        "24d": "Hidden_Dim_Scaling_No_Rotary_123/24d_no_rotary_123.csv",
        "32d": "Hidden_Dim_Scaling_No_Rotary_123/32d_no_rotary_123.csv",
        "64d": "Hidden_Dim_Scaling_No_Rotary_123/64d_no_rotary_123.csv",
        "96d": "Hidden_Dim_Scaling_No_Rotary_123/96d_no_rotary_123.csv",
    },
    "sgd_experiments": {
        "16d": "Hidden_Dim_Scaling_SGD/16d_123_sgd.csv",
        "24d": "Hidden_Dim_Scaling_SGD/24d_123_sgd.csv",
        "32d": "Hidden_Dim_Scaling_SGD/32d_123_sgd.csv",
        "48d": "Hidden_Dim_Scaling_SGD/48d_123_sgd.csv",
        "64d": "Hidden_Dim_Scaling_SGD/64d_123_sgd.csv",
    },
    "lstm_experiments": {
        "16d": "LSTM_Hidden_Dim_Scaling/LSTM_16d_123.csv",
        "24d": "LSTM_Hidden_Dim_Scaling/LSTM_24d_123.csv",
        "32d": "LSTM_Hidden_Dim_Scaling/LSTM_32d_123.csv",
        "48d": "LSTM_Hidden_Dim_Scaling/LSTM_48d_123.csv",
    },
}

CONSTANTS = {"irreducible_loss": 1.76, "compute_effect": 0.154}


# %%
# Power law function
def power_law(x, a, b):
    return a * np.power(x, b)


class ScalingAnalyzer:
    def __init__(self, irreducible_loss=1.76):
        self.irreducible_loss = irreducible_loss

    def extract_scaling_data(self, experiment_group, metric="validation_loss"):
        """Extract final metrics for scaling analysis"""
        losses = []
        compute = []
        configs = []
        for config, exp_data in experiment_group.items():
            losses.append(exp_data["final_metrics"][metric])
            compute.append(exp_data["final_metrics"]["total_flops_profiler"])
            configs.append(config)
        return np.array(compute), np.array(losses), configs

    def fit_power_law(self, compute, losses):
        """Fit power law with irreducible loss adjustment"""
        adjusted_losses = losses - self.irreducible_loss
        # Filter out negative values
        valid_mask = adjusted_losses > 0
        if not np.any(valid_mask):
            return None, None
        try:
            params, _ = curve_fit(
                power_law, compute[valid_mask], adjusted_losses[valid_mask]
            )
            return params
        except:
            return None, None

    def plot_scaling_law(self, compute, losses, label, color=None):
        """Unified plotting function"""
        adjusted_losses = losses - self.irreducible_loss
        valid_mask = adjusted_losses > 0

        if not np.any(valid_mask):
            print(f"Warning: No valid data for {label}")
            return None, None

        compute_valid = compute[valid_mask]
        losses_valid = adjusted_losses[valid_mask]

        params = self.fit_power_law(compute, losses)
        if params is None or params[0] is None:
            print(f"Warning: Could not fit power law for {label}")
            return None, None

        a, b = params

        # Plot data points
        plt.scatter(
            compute_valid, losses_valid, label=f"{label} Data", color=color, alpha=0.7
        )

        # Plot fit line
        x_fit = np.logspace(
            np.log10(min(compute_valid)), np.log10(max(compute_valid)), 100
        )
        y_fit = power_law(x_fit, a, b)
        plt.plot(
            x_fit, y_fit, "--", label=f"{label}: y = {a:.2e} * x^({b:.4f})", color=color
        )

        print(f"{label} - Power law fit: y = {a:.4e} * x^({b:.4f})")
        return a, b


# %%
# Load all experiments
exp_manager = ExperimentManager()
experiments = {}

for group_name, configs in EXPERIMENT_CONFIGS.items():
    experiments[group_name] = exp_manager.load_experiment_group(group_name, configs)

analyzer = ScalingAnalyzer()


# %%
# Create the main scaling law plot
def create_scaling_law_plot():
    plt.figure(figsize=(14, 10))

    colors = ["blue", "red", "green", "orange", "purple"]
    scaling_results = {}

    for i, (group_name, group_data) in enumerate(experiments.items()):
        if len(group_data) == 0:
            print(f"Skipping {group_name} - no data loaded")
            continue

        compute, losses, configs = analyzer.extract_scaling_data(group_data)
        if len(compute) == 0:
            continue

        color = colors[i % len(colors)]
        params = analyzer.plot_scaling_law(compute, losses, group_name, color)
        scaling_results[group_name] = {
            "params": params,
            "compute": compute,
            "losses": losses,
            "configs": configs,
        }

    plt.xlabel("Compute (FLOPS)")
    plt.ylabel("Validation Loss (Irreducible)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("Universal Scaling Laws Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig("Figures/universal_scaling_law_study.png", dpi=300, bbox_inches="tight")
    plt.show()

    return scaling_results


# Run the analysis
scaling_results = create_scaling_law_plot()

# %%
# Print summary results
print("\n=== SCALING LAW SUMMARY ===")
for group_name, results in scaling_results.items():
    if results["params"] and results["params"][0] is not None:
        a, b = results["params"]
        print(f"\n{group_name.upper()}:")
        print(f"  Power law: L = {a:.2e} * C^({b:.4f})")
        print(f"  Configs: {results['configs']}")
        print(
            f"  Loss range: {results['losses'].min():.3f} - {results['losses'].max():.3f}"
        )
        print(
            f"  Compute range: {results['compute'].min():.2e} - {results['compute'].max():.2e}"
        )


# %%
# Compute multiplier analysis
def analyze_compute_multipliers():
    """Analyze compute multipliers between standard and no-rotary experiments"""

    if (
        "hidden_dim_scaling" not in experiments
        or "hidden_dim_no_rotary_123" not in experiments
    ):
        print("Required experiment groups not found for compute multiplier analysis")
        return

    standard_exp = experiments["hidden_dim_scaling"]
    no_rotary_exp = experiments["hidden_dim_no_rotary_123"]

    def compute_multiplier(loss_1, loss_2, irreducible=1.76, compute_effect=0.154):
        diff = np.log(loss_1 - irreducible) - np.log(loss_2 - irreducible)
        return np.exp(diff / compute_effect)

    multipliers = []
    compute_values = []
    config_names = []

    # Find common configurations
    common_configs = set(standard_exp.keys()) & set(no_rotary_exp.keys())

    print("\n=== COMPUTE MULTIPLIER ANALYSIS ===")
    for config in sorted(common_configs):
        loss_no_rotary = no_rotary_exp[config]["final_metrics"]["validation_loss"]
        loss_standard = standard_exp[config]["final_metrics"]["validation_loss"]
        compute_used = standard_exp[config]["final_metrics"]["total_flops_profiler"]

        multiplier = compute_multiplier(loss_no_rotary, loss_standard)
        multipliers.append(multiplier)
        compute_values.append(compute_used)
        config_names.append(config)

        print(
            f"{config:>4}: Standard={loss_standard:.3f}, No-Rotary={loss_no_rotary:.3f}, Multiplier={multiplier:.2f}x"
        )

    if multipliers:
        plt.figure(figsize=(10, 6))
        plt.scatter(compute_values, multipliers, s=100, alpha=0.7)

        # Add labels for each point
        for i, config in enumerate(config_names):
            plt.annotate(
                config,
                (compute_values[i], multipliers[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        plt.xlabel("Compute Used (FLOPS)")
        plt.ylabel("Compute Multiplier (Rotary Advantage)")
        plt.xscale("log")
        plt.title("Rotary Position Encoding Compute Advantage vs Scale")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="No advantage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            "Figures/compute_multiplier_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()


analyze_compute_multipliers()

# %%
