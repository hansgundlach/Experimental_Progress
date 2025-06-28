#!/bin/bash
#SBATCH --job-name=lstm_job_array
#SBATCH --output=slurm-lstm-array-%A_task-%a.out
#SBATCH --error=slurm-lstm-array-%A_task-%a.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=32G
#SBATCH --array=0-3

# Create logs directory name with timestamp
LOG_DIR="logs/lstm_runs/$(date +%d-%H)"
mkdir -p "$LOG_DIR"

# Move SLURM output files to log directory
trap 'mv slurm-lstm-array-${SLURM_ARRAY_JOB_ID}_task-${SLURM_ARRAY_TASK_ID}.* "$LOG_DIR/"' EXIT

echo "Job array ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} started at $(date)"
echo "Working directory: $PWD"
echo "Log directory: $LOG_DIR"

# Clear modules and load required ones
module purge
# module load anaconda/2023a-pytorch
conda activate flashconda

# Print GPU allocation info
echo "GPU allocation for this task:"
nvidia-smi || echo "nvidia-smi not found"

# Run the main experiment, passing the array info
TOTAL_JOBS=${SLURM_ARRAY_TASK_COUNT:-4} # Default to 4 if not set
echo "Running experiment slice ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}..."
python lstm_experiments.py --job_id ${SLURM_ARRAY_TASK_ID} --total_jobs ${TOTAL_JOBS} 2>&1 | tee "$LOG_DIR/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.log"

echo "Job task ${SLURM_ARRAY_TASK_ID} ended at $(date)" 