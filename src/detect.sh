#!/bin/bash -login
#SBATCH --gres=gpu:1
#SBATCH --partition=awd
#SBATCH --mem=16gb
#SBATCH --time=12:00:00

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

python3 /user/home/qy18694/ATI-Pilot-Project/src/Coordinator.py --process_ID=1
