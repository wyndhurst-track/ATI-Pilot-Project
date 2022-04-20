#!/bin/sh -login
### Job name
#PBS -N RGB_latent_0-2
### Output files
#PBS -o RGB_console_output_0-2.stdout
#PBS -e RGB_error_output_0-2.stderr
### Configuration (select chooses the number of nodes)
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environments
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

# Enter the correct directory
cd /home/ca0513/ATI-Pilot-Project/src/Identifier/MetricLearning

# Running parameters
OUT_PATH="/work/ca0513/models/RGBDCows2020/LatentSpace/RGB"
# OUT_PATH="/work/ca0513/models/RGBDCows2020/LatentSpace/RGB-no-difficult"

# Do the main stuff
python train.py --out_path=$OUT_PATH --img_type="RGB" --start_fold=0 --end_fold=2 #--exclude_difficult=1

# Wait for background jobs to complete.
wait
