#!/bin/bash
#SBATCH --job-name=lr_sweep
#SBATCH --output=slurm-lr-%A_task-%a.out
#SBATCH --error=slurm-lr-%A_task-%a.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=32G

# Array specification is set by submit_lr.sh
# GROUP_NAME is passed as the first argument via submit_lr.sh

GROUP_NAME="${1:?Error: GROUP_NAME not provided}"

# Create logs directory
LOG_DIR="logs/lr_sweep_$(date +%d-%H)"
mkdir -p "$LOG_DIR"

# Move SLURM output files to log directory on exit
trap 'mv slurm-lr-${SLURM_ARRAY_JOB_ID}_task-${SLURM_ARRAY_TASK_ID}.* "$LOG_DIR/" 2>/dev/null' EXIT

echo "LR Sweep job ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} started at $(date)"
echo "Group: ${GROUP_NAME}"
echo "Working directory: $PWD"

# Setup environment
module purge
conda activate llm_training

nvidia-smi || echo "nvidia-smi not found"

# Calculate total jobs
if [[ -n "${SLURM_ARRAY_TASK_COUNT}" ]]; then
    TOTAL_JOBS=${SLURM_ARRAY_TASK_COUNT}
elif [[ -n "${SLURM_ARRAY_TASK_MAX}" && -n "${SLURM_ARRAY_TASK_MIN}" ]]; then
    TOTAL_JOBS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
else
    TOTAL_JOBS=1
fi

echo "Running LR sweep slice ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}..."

# Run the LR sweep from the learning_rate_analysis directory
cd learning_rate_analysis || { echo "Failed to cd into learning_rate_analysis"; exit 1; }

python run_lr_sweep.py \
    --group "${GROUP_NAME}" \
    --job_id "${SLURM_ARRAY_TASK_ID}" \
    --total_jobs "${TOTAL_JOBS}" \
    2>&1 | tee "../$LOG_DIR/lr_sweep_${GROUP_NAME}_task_${SLURM_ARRAY_TASK_ID}.log"

echo "Task ${SLURM_ARRAY_TASK_ID} ended at $(date)"
