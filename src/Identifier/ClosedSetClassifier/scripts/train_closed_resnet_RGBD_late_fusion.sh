#!/bin/sh -login
### Job name
#PBS -N RGBD_late_fusion
### Output files
#PBS -o RGBD_late_fusion_console_output.stdout
#PBS -e RGBD_late_fusion_error_output.stderr
### Configuration (select chooses the number of nodes)
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environments
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

# Enter the correct directory
cd /home/ca0513/ATI-Pilot-Project/src/Identifier/ClosedSetClassifier

# Running parameters
OUT_PATH="/work/ca0513/models/RGBDCows2020/ClosedSet/RGBD_late_fusion/"

# Do the main stuff
python ResNet50.py --out_path=$OUT_PATH --img_type="RGBD" --fusion_method="late" --start_fold=8

# Wait for background jobs to complete.
wait
