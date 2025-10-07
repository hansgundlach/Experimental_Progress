"""
Quick diagnostic script to verify experiment configurations are different.
Run this to check if your modern vs 2018 configs are actually different.
"""

from experiment_utils import gen_experim, get_base_config
import json

# Recreate the configs
config_2018 = gen_experim(
    64,
    label="radford_64transformer_2018_bs64",
    folder_name="new_variations",
    learning_rate=10**-2.5,
    activation="gelu",
    norm_placement="post",
    lr_schedule="linear_warmup",
    pos_encoding="learned",
    weight_decay=0.01,
    dropout=0.0,
    optimizer="adam",
    modern_bias_0=False,
    ff_ratio=4,
)[0]["subexperiments"][0]["overrides"]

config_modern_rms_ff4 = gen_experim(
    64,
    label="stanford_modern_rms_ff4",
    folder_name="new_variations",
    learning_rate=10**-2,
    modern_bias_0=True,
    norm_type="rms",
    ff_ratio=4,
)[0]["subexperiments"][0]["overrides"]

config_modern_no_rms_ff4 = gen_experim(
    64,
    label="stanford_modern_no_rms_ff4",
    folder_name="new_variations",
    learning_rate=10**-2,
    modern_bias_0=True,
    ff_ratio=4,
)[0]["subexperiments"][0]["overrides"]

# Get base config for defaults
base_config = get_base_config()


# Merge with base config to see full configs
def merge_with_base(overrides):
    full_config = base_config.copy()
    full_config.update(overrides)
    return full_config


full_2018 = merge_with_base(config_2018)
full_modern_rms = merge_with_base(config_modern_rms_ff4)
full_modern_no_rms = merge_with_base(config_modern_no_rms_ff4)

# Compare key parameters
key_params = [
    "activation",
    "norm_placement",
    "norm_type",
    "pos_encoding",
    "lr_schedule",
    "optimizer",
    "modern_bias_0",
    "ff_ratio",
    "learning_rate",
    "weight_decay",
    "hidden_dim",
    "num_layers",
    "num_heads",
]

print("=" * 80)
print("CONFIGURATION COMPARISON")
print("=" * 80)
print(f"{'Parameter':<20} {'2018':<25} {'Modern+RMS':<25} {'Modern+Layer':<25}")
print("-" * 80)

for param in key_params:
    val_2018 = full_2018.get(param, "NOT SET")
    val_modern_rms = full_modern_rms.get(param, "NOT SET")
    val_modern_no_rms = full_modern_no_rms.get(param, "NOT SET")

    # Highlight if different
    marker = ""
    if val_2018 != val_modern_rms or val_2018 != val_modern_no_rms:
        marker = " <--"

    print(
        f"{param:<20} {str(val_2018):<25} {str(val_modern_rms):<25} {str(val_modern_no_rms):<25}{marker}"
    )

print("=" * 80)

# Calculate parameter counts
from experiment_utils import calculate_transformer_params

params_2018 = calculate_transformer_params(
    full_2018["hidden_dim"],
    full_2018["num_layers"],
    pos_encoding=full_2018["pos_encoding"],
    tie_embeddings=full_2018.get("tie_embeddings", True),
    ff_ratio=full_2018["ff_ratio"],
)

params_modern_rms = calculate_transformer_params(
    full_modern_rms["hidden_dim"],
    full_modern_rms["num_layers"],
    pos_encoding=full_modern_rms["pos_encoding"],
    tie_embeddings=full_modern_rms.get("tie_embeddings", True),
    ff_ratio=full_modern_rms["ff_ratio"],
)

print(f"\nTotal Parameters:")
print(f"  2018:        {params_2018:,}")
print(f"  Modern+RMS:  {params_modern_rms:,}")
print(f"  Difference:  {params_modern_rms - params_2018:,}")

print("\n" + "=" * 80)
print("KEY DIFFERENCES SUMMARY:")
print("=" * 80)

differences = []
for param in key_params:
    val_2018 = full_2018.get(param, "NOT SET")
    val_modern = full_modern_rms.get(param, "NOT SET")
    if val_2018 != val_modern:
        differences.append(f"  • {param}: {val_2018} → {val_modern}")

for diff in differences:
    print(diff)

print("\n" + "=" * 80)
