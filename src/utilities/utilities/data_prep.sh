#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12
#$ -l h_rt=48:00:0




module load anaconda3/2023.03
module load cuda/11.8.0
conda activate hgann
gpu-usage

#export TORCH_HOME=./
#export CUDA_LAUNCH_BLOCKING=1
#export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Declare variables 
python data_prep_audioset.py