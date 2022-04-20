#!/bin/sh -login
### Job name
#PBS -N RTL_08
### Output files
#PBS -o console_output_08.stdout
#PBS -e error_output_08.stderr
### Configuration
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=23gb

# Load the modules/environments
module purge
module load lang/python/anaconda/3.7-2019.10
source activate cows

# Print the job's working directory and enter it.
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Print some other environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
NODES=`cat $PBS_NODEFILE`
echo $NODES
echo PBS job ID is $PBS_JOBID

# Run parameters
UNKNOWN_RATIO=0.8
N=10
OUT_PATH="/work/ca0513/results/CEiA/RTL/08"

# Change directory
cd /home/ca0513/ATI-Pilot-Project/src/Identifier/Supervised-Triplet-Network/

# Do the main stuff
python3 train.py --unknown_ratio=$UNKNOWN_RATIO --out_path=$OUT_PATH --num_repeats=$N

# Wait for background jobs to complete.
wait
