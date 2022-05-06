#!/bin/bash -login
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_short
#SBATCH --mem=16gb
#SBATCH --time=6:00:00

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

python3 /user/home/qy18694/ATI-Pilot-Project/src/Coordinator.py --process_ID=8
