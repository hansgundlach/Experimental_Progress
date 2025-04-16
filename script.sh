#!/bin/bash

#SBATCH -o log-%j
#SBATCH --partition=xeon-g6-volta 
#SBATCH --gres=gpu:volta:1

# Loading the required module

module load anaconda/2023a
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a-pytorch/etc/profile.d/conda.sh
conda init bash

conda activate myenv

python main_experiment.py