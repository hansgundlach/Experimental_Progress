#!/bin/bash
#SBATCH --job-name=lr_analyze
#SBATCH --output=slurm-lr-analyze-%j.out
#SBATCH --error=slurm-lr-analyze-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --mem=8G
# No GPU needed for analysis

GROUP_NAME="${1:?Error: GROUP_NAME not provided}"

echo "LR Analysis for group '${GROUP_NAME}' started at $(date)"

module purge
conda activate llm_training

cd learning_rate_analysis || { echo "Failed to cd into learning_rate_analysis"; exit 1; }

python analyze_lr.py --group "${GROUP_NAME}" 2>&1 | tee "plots/${GROUP_NAME}/analysis_output.log"

echo "Analysis complete at $(date)"
echo "Check plots/${GROUP_NAME}/ for results."
