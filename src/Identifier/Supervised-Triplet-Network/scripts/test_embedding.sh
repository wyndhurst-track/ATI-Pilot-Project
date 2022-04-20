#!/bin/sh 
### Job name
# PBS -N train_cattle_embedding
### Output files
#PBS -o console_output.stdout
#PBS -e error_output.stderr
### Configuration (select chooses the number of nodes)
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true

module load lang/python/anaconda/3.7-2019.03
module load lang/cuda
module load lang/python/anaconda/pytorch

# Print the job's working directory and enter it.
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Print some other environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
NODES=`cat $PBS_NODEFILE`
echo $NODES

# Do the main stuff
python3 test/test_embeddings.py --ckpt_path=/home/ca0513/ATI-Pilot-Project/src/Identifier/Supervised-Triplet-Network/triplet_cnn_cow_id_best_x1.pkl

# Wait for background jobs to complete.
wait
