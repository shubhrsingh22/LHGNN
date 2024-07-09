#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=8G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y
#$ -l node_type=rdg
module load anaconda3/2023.03
module load cuda/11.8.0
conda activate hgann
gpu-usage
#export TORCH_HOME=./
export CUDA_LAUNCH_BLOCKING=1

HYDRA_FULL_ERROR=1 python src/train.py