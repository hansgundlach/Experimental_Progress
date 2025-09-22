#!/usr/bin/env python3
"""
Test script to demonstrate the new colormap color specification functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
from better_looking_graph import TrainingCurveAnalyzer


def test_colormap_colors():
    """Test the colormap color specification feature."""

    # Create a simple test analyzer
    analyzer = TrainingCurveAnalyzer()

    # Test different color specifications
    test_colors = [
        "red",  # Standard color
        "tab:blue",  # Tab color
        "viridis[0]",  # Viridis at index 0
        "viridis[0.5]",  # Viridis at 0.5
        "viridis[1]",  # Viridis at index 1
        "plasma[0.3]",  # Plasma at 0.3
        "inferno[0.7]",  # Inferno at 0.7
        "coolwarm[0.2]",  # Coolwarm at 0.2
    ]

    print("Testing colormap color specifications:")
    for color_spec in test_colors:
        parsed_color = analyzer._parse_color_spec(color_spec)
        print(f"  {color_spec:15} -> {parsed_color}")

    # Create a simple plot to visualize the colors
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 10, 100)
    for i, color_spec in enumerate(test_colors):
        parsed_color = analyzer._parse_color_spec(color_spec)
        y = np.sin(x + i * 0.5)
        ax.plot(x, y, color=parsed_color, label=color_spec, linewidth=2)

    ax.set_title("Colormap Color Specification Test")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("colormap_test.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'colormap_test.png'")
    plt.show()


if __name__ == "__main__":
    test_colormap_colors()
