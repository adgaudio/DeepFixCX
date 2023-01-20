#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=41
#SBATCH -t 24:00:00
#SBATCH -p RM-shared


#type 'man sbatch' for more information and options #SBATCH -p GPU-shared #SBATCH --gpus=1
#this job will ask for 1 V100 GPUs in GPU-shared for 1 second

#echo commands to stdout

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

module load AI/anaconda3-tf2.2020.11
source /opt/packages/anaconda3/2020.11/etc/profile.d/conda.sh
conda deactivate
conda activate waveletfix
cd ~/store/waveletfix
export PYTHONPATH=`pwd`:$PYTHONPATH

#run pre-compiled program which is already in your project space

exec ./bin/experiments_bridges.sh
exit  # probably not needed
