#!/bin/bash

#SBATCH -o logs/%d-%H/%j.log
#SBATCH --partition=xeon-g6-volta 
#SBATCH --gres=gpu:volta:2

# Create logs directory with timestamp
mkdir -p logs/$(date +%d-%H)

# Loading the required module
module load anaconda/2023a
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh
conda init bash

conda activate myenv

python main_experiment.py