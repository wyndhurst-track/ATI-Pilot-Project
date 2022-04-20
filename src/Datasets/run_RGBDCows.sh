#!/bin/sh -login
### Job name
#PBS -N RGBDCows
### Output files
#PBS -o console_output.stdout
#PBS -e error_output.stderr
### Configuration (select chooses the number of nodes)
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environments
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

# Enter the correct directory
cd /home/ca0513/ATI-Pilot-Project/src/Datasets

# Do the main stuff
python RGBDCows2020.py

# Wait for background jobs to complete.
wait
