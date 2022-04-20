#!/bin/sh -login
### Job name
#PBS -N tvt_D
### Output files
#PBS -o tvt_D_console_output.stdout
#PBS -e tvt_D_error_output.stderr
### Configuration (select chooses the number of nodes)
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environments
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

# Enter the correct directory
cd /home/ca0513/ATI-Pilot-Project/src/Identifier/MetricLearning

### Running parameters

# Where to store outputs
OUT_PATH="/work/ca0513/models/RGBDCows2020/LatentSpace/train_valid_test/D"
OUT_PATH_NO_DIFFICULT="/work/ca0513/models/RGBDCows2020/LatentSpace/train_valid_test/D-no-difficult"

# How many folds to run for (just a single one in this case)
NUM_FOLDS=1

# Train with difficult animals included
python train.py --out_path=$OUT_PATH --img_type="D" --num_folds=$NUM_FOLDS

# Train with difficult animals removed
python train.py --out_path=$OUT_PATH_NO_DIFFICULT --img_type="D" --num_folds=$NUM_FOLDS --exclude_difficult=1

# Wait for background jobs to complete.
wait
