#!/bin/bash
#SBATCH --job-name=exp_job_array
#SBATCH --output=slurm-array-%A_task-%a.out
#SBATCH --error=slurm-array-%A_task-%a.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=16G

# Create logs directory name with timestamp
LOG_DIR="logs/$(date +%d-%H)"
mkdir -p "$LOG_DIR"

# Move SLURM output files to log directory at the end of each task
trap 'mv slurm-array-${SLURM_ARRAY_JOB_ID}_task-${SLURM_ARRAY_TASK_ID}.* "$LOG_DIR/"' EXIT

echo "Job array ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} started at $(date)"
echo "Working directory: $PWD"
echo "Log directory: $LOG_DIR"

# Environment
module purge
# module load anaconda/2023a-pytorch  # Uncomment if your cluster uses module for conda
conda activate llm_training || echo "Warning: conda env 'llm_training' not found; proceeding with current env"

# GPU info
echo "GPU allocation for this task:"
nvidia-smi || echo "nvidia-smi not found"

# Total jobs = array size
TOTAL_JOBS=${SLURM_ARRAY_TASK_COUNT:-1}
JOB_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Running experiment slice ${JOB_ID} of ${TOTAL_JOBS}..."
python experiments.py --job_id ${JOB_ID} --total_jobs ${TOTAL_JOBS} 2>&1 | tee "$LOG_DIR/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.log"

echo "Job task ${SLURM_ARRAY_TASK_ID} ended at $(date)"
