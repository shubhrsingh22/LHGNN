#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 32
#$ -l h_rt=6:30:00
#$ -l h_vmem=2G
#$ -l gpu=4

module load anaconda3/2023.03
module load cuda/11.8.0
conda activate hgann
gpu-usage

export TORCH_HOME=./
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Declare variables 
max_epochs=50
min_epochs=50
num_devices=4
batch_size=32
strategy='ddp'
num_nodes=1
# Testing with one epoch 
HYDRA_FULL_ERROR=1 python src/train.py  trainer.max_epochs=$max_epochs trainer.min_epochs=$min_epochs trainer.num_nodes=$num_nodes trainer.devices=$num_devices trainer.strategy=$strategy data.batch_size=$((batch_size * 1))