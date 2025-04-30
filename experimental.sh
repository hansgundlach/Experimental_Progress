#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=32G

# Create logs directory name with timestamp
LOG_DIR="logs/$(date +%d-%H)"
mkdir -p "$LOG_DIR"

# Move SLURM output files to log directory
trap 'mv slurm-${SLURM_JOB_ID}.* "$LOG_DIR/"' EXIT

# Create logs directory name with timestamp
LOG_DIR="logs/$(date +%d-%H)"
mkdir -p $LOG_DIR

echo "Job started at $(date)"
echo "Working directory: $PWD"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Log directory: $LOG_DIR"

# Clear modules and load required ones
module purge
# module load anaconda/2023a-pytorch
conda activate flashconda

# Print GPU allocation info
echo "GPU allocation:"
nvidia-smi || echo "nvidia-smi not found"

# Create a simple GPU check script
cat > check_gpu.py << 'EOF'
import torch
import sys
import os

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ERROR: CUDA is not available!")
    sys.exit(1)
EOF

# Check GPU availability
echo "Checking GPU availability..."
python check_gpu.py || exit 1

# Run the main experiment
echo "Running main experiment..."
python main_experiment.py --device cuda:0 2>&1 | tee "$LOG_DIR/$SLURM_JOB_ID.log"

echo "Job ended at $(date)"