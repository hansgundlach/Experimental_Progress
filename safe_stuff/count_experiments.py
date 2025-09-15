#!/usr/bin/env python3
"""Count the total number of experiments defined in experiments.py"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def count_experiments():
    # Read experiments.py to find the EXPERIMENTS assignment
    experiments_file = os.path.join(os.path.dirname(__file__), "experiments.py")

    with open(experiments_file, "r") as f:
        content = f.read()

    # Find the line that sets EXPERIMENTS =
    lines = content.split("\n")
    experiments_line = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EXPERIMENTS = ") and not stripped.startswith(
            "# EXPERIMENTS"
        ):
            experiments_line = stripped
            break

    if not experiments_line:
        raise ValueError(
            "Could not find uncommented 'EXPERIMENTS = ' assignment in experiments.py"
        )

    # Extract the variable name(s) after the equals sign
    rhs = experiments_line.split("EXPERIMENTS = ")[1].strip()

    # Import ALL experiment definitions so eval can work with any combination
    import experiment_definitions

    # Create a namespace with all the experiment variables
    namespace = {}
    for attr in dir(experiment_definitions):
        if not attr.startswith("_"):  # Skip private attributes
            namespace[attr] = getattr(experiment_definitions, attr)

    # Also add any functions that might be used (like subset_experiments, create_multi_lr_experiments)
    # Import from experiments.py itself
    import experiments

    for attr in dir(experiments):
        if not attr.startswith("_") and callable(getattr(experiments, attr)):
            namespace[attr] = getattr(experiments, attr)

    # Evaluate the right-hand side to get the actual experiments list
    try:
        actual_experiments = eval(rhs, namespace)
    except Exception as e:
        raise ValueError(f"Unable to evaluate EXPERIMENTS assignment '{rhs}': {e}")

    # Count sub-experiments
    total = 0
    for exp in actual_experiments:
        total += len(exp["subexperiments"])

    return total


if __name__ == "__main__":
    print(count_experiments())
