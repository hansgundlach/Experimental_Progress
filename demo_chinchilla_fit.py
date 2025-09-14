#!/usr/bin/env python3
"""
Demonstration script showing how to use the new Chinchilla-style fitting option.
"""

import sys
import os

# Add the experimental_analysis directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "experimental_analysis"))

from fit_hitchhikers_loss import (
    fit_validation_loss_from_pairs,
    get_dataset_configurations,
)


def demo_chinchilla_fitting():
    """Demonstrate the new Chinchilla-style fitting functionality."""

    print("Chinchilla-Style Fitting Demo")
    print("=" * 50)

    # Get available dataset configurations
    configurations = get_dataset_configurations()

    if not configurations:
        print(
            "No dataset configurations found. Please check your experimental_data_folder."
        )
        return

    # Use the first available configuration
    dataset_name = list(configurations.keys())[0]
    files_and_N = configurations[dataset_name]

    if not files_and_N:
        print(f"No files found for dataset: {dataset_name}")
        return

    print(f"Using dataset: {dataset_name}")
    print(f"Number of files: {len(files_and_N)}")
    print()

    # Standard fitting (original method)
    print("1. Standard Fitting (Original Method)")
    print("-" * 40)
    try:
        fit_standard = fit_validation_loss_from_pairs(
            files_and_N,
            use_chinchilla_sk_fit=False,
            compute_confidence_intervals=True,
            use_bootstrap=True,
            n_bootstrap=50,  # Reduced for demo
        )

        print(f"Alpha (parameter scaling): {fit_standard.alpha:.4f}")
        print(f"Beta (data scaling): {fit_standard.beta:.4f}")
        print(f"Gamma: {fit_standard.gamma:.4f}")
        print(f"Final MSE loss: {fit_standard.final_loss:.6f}")
        print()

    except Exception as e:
        print(f"Standard fitting failed: {e}")
        return

    # Chinchilla-style fitting
    print("2. Chinchilla-Style Fitting (Huber Loss + Logsumexp)")
    print("-" * 50)
    try:
        fit_chinchilla = fit_validation_loss_from_pairs(
            files_and_N,
            use_chinchilla_sk_fit=True,
            huber_beta=0.1,  # Huber loss delta parameter
            compute_confidence_intervals=True,
            use_bootstrap=True,
            n_bootstrap=50,  # Reduced for demo
        )

        print(f"Alpha (parameter scaling): {fit_chinchilla.alpha:.4f}")
        print(f"Beta (data scaling): {fit_chinchilla.beta:.4f}")
        print(f"Gamma: {fit_chinchilla.gamma:.4f}")
        print(f"Final Huber loss: {fit_chinchilla.final_loss:.6f}")
        print()

    except Exception as e:
        print(f"Chinchilla fitting failed: {e}")
        return

    # Comparison
    print("3. Comparison")
    print("-" * 20)
    print(f"Alpha difference: {abs(fit_standard.alpha - fit_chinchilla.alpha):.4f}")
    print(f"Beta difference: {abs(fit_standard.beta - fit_chinchilla.beta):.4f}")
    print(f"Gamma difference: {abs(fit_standard.gamma - fit_chinchilla.gamma):.4f}")
    print()

    print("Key Differences:")
    print("- Standard fitting uses MSE loss in linear space")
    print("- Chinchilla fitting uses Huber loss in log space with logsumexp model")
    print(
        "- Both methods can produce different results due to different loss functions"
    )
    print("- Chinchilla method is more robust to outliers due to Huber loss")


if __name__ == "__main__":
    demo_chinchilla_fitting()
