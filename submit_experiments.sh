#!/bin/bash

echo "=== Dynamic Experiment Submission Script ==="
echo "Counting experiments..."

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Activate conda environment if available
CONDA_PATH=$(which conda)
if [ -n "$CONDA_PATH" ]; then
    echo "Activating conda environment..."
    source "$(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh"
    
    # Try to activate the environment, but don't fail if it doesn't exist
    conda activate llm_training 2>/dev/null || echo "Note: llm_training environment not found, using base environment"
fi

# Count the experiments
NUM_EXPERIMENTS=$(python count_experiments.py)
if [ $? -ne 0 ]; then
    echo "Error: Failed to count experiments"
    exit 1
fi

# Configuration
MAX_CONCURRENT=8  # Adjust based on your cluster limits/preferences

echo "Found $NUM_EXPERIMENTS experiments"
echo "Will use array 0-$((NUM_EXPERIMENTS-1))%${MAX_CONCURRENT}"

# Validate we have experiments
if [ "$NUM_EXPERIMENTS" -eq 0 ]; then
    echo "Error: No experiments found!"
    exit 1
fi

# Create a temporary main.sh with the correct array size
TEMP_MAIN="main_dynamic.sh"
cp main.sh "$TEMP_MAIN"

# Replace the array line (accounting for 0-indexed arrays)
NEW_ARRAY_LINE="#SBATCH --array=0-$((NUM_EXPERIMENTS-1))%${MAX_CONCURRENT}"

# Use portable sed syntax (works on both Linux and macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS sed requires a backup extension with -i
    sed -i .bak "s|#SBATCH --array=.*|${NEW_ARRAY_LINE}|" "$TEMP_MAIN"
    sed -i .bak 's|echo "GPU allocation for this task:":Q|echo "GPU allocation for this task:"|' "$TEMP_MAIN"
    rm "${TEMP_MAIN}.bak"
else
    # Linux sed
    sed -i "s|#SBATCH --array=.*|${NEW_ARRAY_LINE}|" "$TEMP_MAIN"
    sed -i 's|echo "GPU allocation for this task:":Q|echo "GPU allocation for this task:"|' "$TEMP_MAIN"
fi

echo "Created temporary job script: $TEMP_MAIN"
echo "Array configuration: $NEW_ARRAY_LINE"

# Show what we're about to submit
echo ""
echo "=== Job Configuration ==="
grep "#SBATCH" "$TEMP_MAIN"
echo ""

# Submit the job
echo "Submitting job array..."
sbatch "$TEMP_MAIN"
SUBMIT_RESULT=$?

if [ $SUBMIT_RESULT -eq 0 ]; then
    echo "Job submitted successfully!"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel with: scancel <job_id>"
else
    echo "Error: Job submission failed"
fi

# Clean up temporary file
rm "$TEMP_MAIN"
echo "Cleaned up temporary file: $TEMP_MAIN"

exit $SUBMIT_RESULT
