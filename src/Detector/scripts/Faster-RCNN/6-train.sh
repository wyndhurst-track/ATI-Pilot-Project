#!/bin/sh 
### Job name
#PBS -N frcnn_cows_f6
### Output files
#PBS -o console_output_fold-6.stdout
#PBS -e error_output_fold-6.stderr
### Configuration (select chooses the number of nodes)
#PBS -l walltime=34:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

#PBS -m a

# Load the modules and anaconda environment
module purge
module load lang/python/anaconda/3.7-2019.03-tensorflow
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

# Parameters
FOLD=6
DATA_DIR=/work/ca0513/datasets/CEADetection/

# Change to the base directory containing our script
cd /home/ca0513/ATI-Pilot-Project/src/Detector/

# Run the main stuff
python3 FasterRCNNWrapper.py --fold $FOLD --dataset_loc $DATA_DIR

# Wait for background jobs to complete.
wait
