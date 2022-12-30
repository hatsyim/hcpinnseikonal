#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB 
#SBATCH --time=12:00:00 
#SBATCH --partition=batch 

conda activate my_env_plot

sh Example-5.sh