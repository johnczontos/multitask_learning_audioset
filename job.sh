#!/bin/bash
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH -w cn-m-2
#SBATCH --job-name=data_profile
#SBATCH -t 1-00:00:00
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --export=ALL
#SBATCH -o data_profile.out
#SBATCH -e data_profile.err

module load python/3.12 cuda/12.1

# activate env
source env/bin/activate

python data_profile.py