#!/bin/bash
#SBATCH -J panna
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 01:30:00
#SBATCH -c 32
#SBATCH -p allgroups
#SBATCH --mem 256G

cd $WORKING_DIR

export OMP_NUM_THREADS=32

srun singularity run ./panna.sif python/tests/datasets_test.py