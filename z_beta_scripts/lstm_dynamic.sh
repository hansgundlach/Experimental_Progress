#!/bin/bash

# First, count the experiments
EXP_COUNT=$(python lstm_experiments.py --count_only)
MAX_CONCURRENT=4

# Calculate array range (0 to EXP_COUNT-1)
ARRAY_MAX=$((EXP_COUNT - 1))

echo "Found $EXP_COUNT experiments, setting array range 0-$ARRAY_MAX with max $MAX_CONCURRENT concurrent"

# Submit the job with dynamic array range
sbatch --array=0-${ARRAY_MAX}%${MAX_CONCURRENT} << 'EOF'
#!/bin/bash
#SBATCH --job-name=lstm_job_array
#SBATCH --output=slurm-lstm-array-%A_task-%a.out
#SBATCH --error=slurm-lstm-array-%A_task-%a.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=32G

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
if [[ -n "${SLURM_ARRAY_TASK_COUNT}" ]]; then
TOTAL_JOBS=${SLURM_ARRAY_TASK_COUNT}
elif [[ -n "${SLURM_ARRAY_TASK_MAX}" && -n "${SLURM_ARRAY_TASK_MIN}" ]]; then
TOTAL_JOBS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
else
TOTAL_JOBS=1
fi
echo "Running experiment slice ${SLURM_ARRAY_TASK_ID} of ${TOTAL_JOBS}..."
python lstm_experiments.py --job_id ${SLURM_ARRAY_TASK_ID} --total_jobs ${TOTAL_JOBS} 2>&1 | tee "$LOG_DIR/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.log"

echo "Job task ${SLURM_ARRAY_TASK_ID} ended at $(date)"
EOF

