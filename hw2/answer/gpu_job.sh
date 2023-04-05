#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 04:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -A see230002p

source /jet/home/zhen1997/miniconda3/etc/profile.d/conda.sh
conda activate auto3D

# go to your working directory
cd '/jet/home/zhen1997/teaching/HW2'
python vgg16_demo.py
