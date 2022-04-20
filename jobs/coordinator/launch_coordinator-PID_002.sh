#!/bin/sh -login
### Job name
#PBS -N ATI_COORD_002
# Name the files it will output
#PBS -o ATI_COORD_002_console_output.stdout
#PBS -e ATI_COORD_002_error_output.stderr
### Job configuration
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate cows

# Enter the correct directory
cd /home/qy18694/ATI-Pilot-Project/src/

# Command line arugments
PROCESS_ID=2

# Run it
python Coordinator.py --process_ID=$PROCESS_ID

# Wait for background jobs to complete.
wait
