#!/bin/bash
#SBATCH --job-name=lstm_single_gpu
#SBATCH --output=LSTM_model/logs/slurm-lstm-array-%A_task-%a.out
#SBATCH --error=LSTM_model/logs/slurm-lstm-array-%A_task-%a.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=128G
# Array specification is now handled by submit_lstm_job.sh wrapper

# Create logs directory
LOG_DIR="LSTM_model/logs"
mkdir -p "$LOG_DIR"

echo "Job array ${SLURM_ARRAY_JOB_ID}, Task ${SLURM_ARRAY_TASK_ID} started at $(date)"
echo "Working directory: $PWD"
echo "Log directory: $LOG_DIR"

# Clear modules and load required ones
module purge
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh
conda activate llm_training

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
# The job starts in the main project directory, so we need to go into LSTM_model
echo "Current directory: $PWD"
echo "Contents: $(ls -la)"
cd LSTM_model || { echo "Failed to cd into LSTM_model"; exit 1; }
echo "Changed to LSTM_model directory: $PWD"
echo "LSTM_model contents: $(ls -la)"
python lstm_experiments.py --job_id ${SLURM_ARRAY_TASK_ID} --total_jobs ${TOTAL_JOBS} 2>&1 | tee "logs/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.log"

echo "Job task ${SLURM_ARRAY_TASK_ID} ended at $(date)" 