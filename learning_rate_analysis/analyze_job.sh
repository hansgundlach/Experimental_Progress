#!/bin/bash
#SBATCH --job-name=lr_analyze
#SBATCH --output=learning_rate_analysis/logs/slurm-lr-analyze-%j.out
#SBATCH --error=learning_rate_analysis/logs/slurm-lr-analyze-%j.err
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --mem=8G
# No GPU needed for analysis — runs on the CPU-only partition.

GROUP_NAME="${1:?Error: GROUP_NAME not provided}"

echo "LR Analysis for group '${GROUP_NAME}' started at $(date)"

module purge
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh
conda activate llm_training

# Ensure analysis deps are available (the llm_training env may not include them).
# pip --user writes to ~/.local, which is persistent across nodes.
python -c "import matplotlib, pandas, numpy" 2>/dev/null || \
    pip install --user --quiet matplotlib pandas numpy

cd learning_rate_analysis || { echo "Failed to cd into learning_rate_analysis"; exit 1; }

mkdir -p "plots/${GROUP_NAME}"
python analyze_lr.py --group "${GROUP_NAME}" 2>&1 | tee "plots/${GROUP_NAME}/analysis_output.log"

echo "Analysis complete at $(date)"
echo "Check plots/${GROUP_NAME}/ for results."
