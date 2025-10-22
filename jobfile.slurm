#!/bin/bash
#SBATCH -J panna
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH --partition lovelace
#SBATCH --mem 0
#SBATCH --exclusive

cd $WORKING_DIR


srun ./panna python/tests/datasets_test.py