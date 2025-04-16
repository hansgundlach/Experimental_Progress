#!/bin/bash
#SBATCH --job-name=wiki-text-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=xeon-p8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load modules
module load python/3.8
module load cuda/11.3

# Activate your environment
source ~/myenv/bin/activate

# (Optional) Set your wandb API key
export WANDB_API_KEY=a280a7020a1069b251f6d58b3d63b1a23fc57802

# Print some info
echo "Running on node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run your experiment
python experimental/architecture_experiments/wiki_text_experiment.py