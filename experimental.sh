#!/bin/bash

# Create logs directory with timestamp
LOG_DIR="logs/$(date +%d-%H)"
mkdir -p $LOG_DIR

#SBATCH -o ${LOG_DIR}/slurm-%j.out
#SBATCH -e ${LOG_DIR}/slurm-%j.err
#SBATCH --partition=xeon-g6-volta 
#SBATCH --gres=gpu:volta:2
#SBATCH --mem=32G

# Add debugging output
echo "Job started at $(date)"
echo "Working directory: $PWD"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Log directory: $LOG_DIR"

# Check SLURM GPU allocation
echo "SLURM GPU allocation:"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"

# Load modules using module purge first to avoid conflicts
echo "Purging and loading modules..."
module purge
# Try loading CUDA (with error handling)
if module load cuda/11.8 2>/dev/null; then
    echo "Successfully loaded CUDA 11.8"
else
    echo "Warning: Failed to load cuda/11.8 module"
    echo "Available CUDA modules:"
    module avail cuda
fi

# Try loading NCCL (with error handling)
if module load nccl/2.18.1-cuda11.8 2>/dev/null; then
    echo "Successfully loaded NCCL 2.18.1"
else
    echo "Warning: Failed to load NCCL module"
    echo "Available NCCL modules:"
    module avail nccl
fi

# Check if CUDA is accessible
echo "Checking for nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi is available:"
    nvidia-smi
else
    echo "Warning: nvidia-smi not found"
fi

# Load anaconda module
# echo "Loading anaconda module..."
# module load anaconda/2023a-pytorch
# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh

# Activate conda environment
echo "Activating conda environment..."
conda activate flashconda
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check if PyTorch can see CUDA
echo "Checking PyTorch CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]"

# Set environment variables to force CUDA usage
echo "Setting up CUDA environment variables..."
export CUDA_VISIBLE_DEVICES="0,1"
# Force PyTorch to use CUDA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Modify main_experiment.py to force CUDA usage
echo "Creating wrapper to force CUDA usage..."
cat > force_cuda_wrapper.py << 'EOF'
import sys
import torch
import os

# Check if CUDA is available before importing main script
print("CUDA availability check in wrapper:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: CUDA is not available!")

# Now override the get_device function in main_experiment.py
original_import = __import__

def patched_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    
    # Override get_device function after main_experiment is imported
    if name == 'main_experiment':
        print("Patching get_device function to force CUDA usage...")
        
        def force_cuda_device(gpu_id=None):
            if torch.cuda.is_available():
                if gpu_id is not None and gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    device = torch.device("cuda:0")
                print(f"Forced CUDA device: {device}")
                return device
            else:
                print("WARNING: CUDA not available despite forcing!")
                return torch.device("cpu")
                
        module.get_device = force_cuda_device
    
    return module

# Override __import__ to patch the function
__builtins__.__import__ = patched_import

# Import and run the main script
print("Importing and running main_experiment.py...")
import main_experiment
EOF

# Run the wrapper script instead of main_experiment.py directly
echo "Running main experiment with CUDA forcing wrapper..."
python force_cuda_wrapper.py 2>&1 | tee "$LOG_DIR/$SLURM_JOB_ID.log"

echo "Job ended at $(date)"