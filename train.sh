#!/bin/bash
#SBATCH -A sayandebroy
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --nodelist gnode030
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00
#SBATCH --output=output.txt

conda init bash
source activate driving

which python

python train.py
